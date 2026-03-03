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

#include "acl/acl_base.h"
#define protected public
#define private public
#include "model/acl_resource_manager.h"
#undef private
#undef protected
#include "acl_stub.h"

#include <vector>

#define protected public
#define private public
#include "single_op/shape_range_utils.h"
#undef private
#undef protected

#define protected public
#define private public
#include "utils/acl_op_map.h"
#include "single_op/compile/op_compile_service.h"
#include "single_op/compile/op_compile_processor.h"
#undef private
#undef protected
#include "acl/acl.h"
#include "acl/acl_rt_allocator.h"
#include "utils/hash_utils.h"
#include "utils/file_utils.h"
#include "utils/attr_utils.h"
#include "single_op/op_model_parser.h"
#include "single_op/compile/op_compile_service.h"
#include "utils/hash_utils.h"
#include "single_op/compile/local_compiler.h"
#include "framework/memory/allocator_desc.h"
#include "ge/ge_allocator.h"

using namespace std;
using namespace testing;
using namespace acl;
using namespace acl::file_utils;
using namespace acl::attr_utils;

class UTEST_ACL_Resource_Manager : public testing::Test {
protected:
    virtual void SetUp() {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
        ON_CALL(MockFunctionTest::aclStubInstance(), GetPlatformResWithLock(_, _))
                .WillByDefault(Return(true));
    }
    virtual void TearDown() {
        // 单例测试，需要恢复初始状态
        acl::AclResourceManager::GetInstance().streamExternalAllocator_.clear();
        acl::AclResourceManager::GetInstance().streamDefaultAllocator_.clear();
        Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));
    }
};

static void *g_priCtx = nullptr;

TEST(UTEST_ACL_Resource_Manager, TestGetEnv) {
    AclResourceManager mng;

    // no env
    int32_t mmRet = 0;
    MM_SYS_UNSET_ENV(MM_ENV_ENABLE_RUNTIME_V2, mmRet);
    (void)mmRet;
    mng.GetRuntimeV2Env();

    bool modelV2Enable = mng.IsRuntimeV2Enable(true);
    bool singleOpV2Enable = mng.IsRuntimeV2Enable(false);
    EXPECT_EQ(modelV2Enable, true);
    EXPECT_EQ(singleOpV2Enable, true);

    // env is 0
    MM_SYS_SET_ENV(MM_ENV_ENABLE_RUNTIME_V2, "0", 1, mmRet);
    (void)mmRet;
    mng.GetRuntimeV2Env();
    modelV2Enable = mng.IsRuntimeV2Enable(true);
    singleOpV2Enable = mng.IsRuntimeV2Enable(false);
    EXPECT_EQ(modelV2Enable, false);
    EXPECT_EQ(singleOpV2Enable, false);

    // env is 1
    MM_SYS_SET_ENV(MM_ENV_ENABLE_RUNTIME_V2, "1", 1, mmRet);
    mng.GetRuntimeV2Env();
    modelV2Enable = mng.IsRuntimeV2Enable(true);
    singleOpV2Enable = mng.IsRuntimeV2Enable(false);
    EXPECT_EQ(modelV2Enable, true);
    EXPECT_EQ(singleOpV2Enable, true);

    // env is empty
    MM_SYS_SET_ENV(MM_ENV_ENABLE_RUNTIME_V2, "", 1, mmRet);
    mng.GetRuntimeV2Env();
    modelV2Enable = mng.IsRuntimeV2Enable(true);
    singleOpV2Enable = mng.IsRuntimeV2Enable(false);
    EXPECT_EQ(modelV2Enable, true);
    EXPECT_EQ(singleOpV2Enable, true);

    // env is 2
    MM_SYS_SET_ENV(MM_ENV_ENABLE_RUNTIME_V2, "2", 1, mmRet);
    mng.GetRuntimeV2Env();
    modelV2Enable = mng.IsRuntimeV2Enable(true);
    singleOpV2Enable = mng.IsRuntimeV2Enable(false);
    EXPECT_EQ(modelV2Enable, true);
    EXPECT_EQ(singleOpV2Enable, false);
}

TEST(UTEST_ACL_Resource_Manager, TestRtSession) {
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    uint32_t modelId = 0x1;
    EXPECT_EQ(acl::AclResourceManager::GetInstance().GetRtSession(modelId), nullptr);

    auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
    acl::AclResourceManager::GetInstance().AddExecutor(modelId, std::move(executor), rtSession);
    EXPECT_EQ(acl::AclResourceManager::GetInstance().GetRtSession(modelId), rtSession);

    EXPECT_EQ(acl::AclResourceManager::GetInstance().DeleteExecutor(modelId), ACL_SUCCESS);
    EXPECT_EQ(acl::AclResourceManager::GetInstance().GetRtSession(modelId), nullptr);
}

TEST(UTEST_ACL_Resource_Manager, RegisterModelTest)
{
    OpModelDef modelDef;
    modelDef.opType = "testOp";
    AclOpResourceManager::ModelMap modelMap;
    auto &instance = AclOpResourceManager::GetInstance();
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), modelMap, false, isDeduplicate), ACL_SUCCESS);
}

aclError ListFilesMock(const std::string &dirName,
                       FileNameFilterFn filter,
                       std::vector<std::string> &names,
                       int maxDepth)
{
    (void) dirName;
    (void) filter;
    (void) maxDepth;
    names.emplace_back("test_model.om");
    return ACL_SUCCESS;
}

aclError ParseOpModelMock(OpModel &opModel, OpModelDef &modelDef)
{
    (void) opModel;
    modelDef.opType = "testOp";
    return ACL_SUCCESS;
}

TEST(UTEST_ACL_Resource_Manager, CheckValueRangeTest)
{
    SetCastHasTruncateAttr(false);
    ge::GeAttrValue tensorValue;
    ge::GeTensor geTensor;

    tensorValue.SetValue<ge::GeTensor>(geTensor);
    std::map<AttrRangeType, ge::GeAttrValue> attrMap;
    attrMap[AttrRangeType::VALUE_TYPE] = tensorValue;

    OpModelDef modelDef;
    modelDef.opType = "test_model";
    int64_t shape[]{16, -1};
    int64_t shapeStatic[]{16, 16};
    int64_t range[2][2] = {{16, 16}, {1, 16}};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    modelDef.inputDescArr[0].valueRange = attrMap;
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef.inputDescArr[0], 2, range);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");
    modelDef.opAttr.SetAttr<string>("truncate", "1");

    OpModelDef modelDef1;
    modelDef1.opType = "test_model";
    int64_t shape1[]{16, -1};
    int64_t shapeStatic1[]{16, 16};
    int64_t range1[2][2] = {{16, 16}, {1, 16}};
    modelDef1.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape1, ACL_FORMAT_ND);

    modelDef1.inputDescArr[0].valueRange = attrMap;
    modelDef1.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic1, ACL_FORMAT_ND);
    modelDef1.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic1, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef1.inputDescArr[0], 2, range1);
    modelDef1.opAttr.SetAttr<string>("testAttr", "attrValue");
    modelDef1.opAttr.SetAttr<string>("truncate", "1");

    auto &instance = AclOpResourceManager::GetInstance();
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, true, isDeduplicate), ACL_SUCCESS);

    EXPECT_EQ(instance.RegisterModel(std::move(modelDef1), instance.opModels_, true, isDeduplicate), ACL_SUCCESS);
}

TEST(UTEST_ACL_Resource_Manager, MatchModelDynamicTest)
{
    SetCastHasTruncateAttr(false);
    OpModelDef modelDef;
    modelDef.opType = "Cast";
    int64_t shape[]{16, -1};
    int64_t shapeStatic[]{16, 16};
    int64_t range[2][2] = {{16, 16}, {1, 16}};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef.inputDescArr[0], 2, range);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");
    modelDef.opAttr.SetAttr<string>("truncate", "1");

    auto &instance = AclOpResourceManager::GetInstance();
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, true, isDeduplicate), ACL_SUCCESS);

    OpModelDef modelDef_2;
    modelDef.opType = "test_acl";
    int64_t shape_2[]{16, -1};
    int64_t shapeStatic_2[]{16, 16};
    int64_t range_2[2][2] = {{16, 16}, {1, 16}};
    modelDef_2.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape_2, ACL_FORMAT_ND);
    modelDef_2.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic_2, ACL_FORMAT_ND);
    modelDef_2.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic_2, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef_2.inputDescArr[0], 2, range_2);
    modelDef_2.opAttr.SetAttr<string>("testAttr", "attrValue");
    modelDef_2.opAttr.SetAttr<string>("truncate", "1");


    AclOp aclOp;
    aclopAttr *opAttr = aclopCreateAttr();
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    int64_t shapeFind[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);

    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;

    OpModel opModel;
    bool isDynamic;
    aclOp.opType = "Cast";
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.numInputs = 1;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.numInputs = 2;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.numOutputs = 1;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.opAttr = opAttr;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclopSetAttrString(opAttr, "testAttr", "invalid");
    const_cast<aclTensorDesc *>(aclOp.outputDesc[0])->memtype = ACL_MEMTYPE_HOST;
    const_cast<aclTensorDesc *>(aclOp.outputDesc[0])->isConst = false;
    aclDataBuffer *outputs[1] = {nullptr};
    char ptr[4] = {0};
    outputs[0] = aclCreateDataBuffer(ptr, 4);
    aclOp.outputs = outputs;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    const_cast<aclTensorDesc *>(aclOp.outputDesc[0])->memtype = ACL_MEMTYPE_DEVICE;
    const_cast<aclTensorDesc *>(aclOp.outputDesc[0])->isConst = false;
    aclOp.outputs = nullptr;
    aclDestroyDataBuffer(outputs[0]);
    aclopSetAttrString(opAttr, "testAttr", "attrValue");
    EXPECT_EQ(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_ERROR_OP_NOT_FOUND);
    aclOp.isCompile = true;
    EXPECT_EQ(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_ERROR_OP_NOT_FOUND);
    aclOp.isCompile = false;
    aclDestroyTensorDesc(inputDesc[0]);
    int64_t shapeFindDynamic[]{16, -1};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFindDynamic, ACL_FORMAT_ND);
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.isCompile = true;
    aclDestroyTensorDesc(inputDesc[0]);
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    int64_t rangeStatic[2][2] = {{16, 16}, {16, 16}};
    aclSetTensorShapeRange((aclTensorDesc*)aclOp.inputDesc[0], 2, range);
    aclSetTensorShapeRange((aclTensorDesc*)aclOp.inputDesc[1], 2, rangeStatic);
    aclSetTensorShapeRange((aclTensorDesc*)aclOp.outputDesc[0], 2, rangeStatic);
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    int64_t range2[2][2] = {{16, 16}, {1, 32}};
    aclSetTensorShapeRange((aclTensorDesc*)aclOp.inputDesc[0], 2, range2);
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    int64_t range3[2][2] = {{16, 16}, {1, -1}};
    aclSetTensorShapeRange((aclTensorDesc*)aclOp.inputDesc[0], 2, range3);
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    int64_t shapeFind2[]{16, 0};
    aclDestroyTensorDesc(inputDesc[0]);
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind2, ACL_FORMAT_ND);
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    int64_t shapeFind3[]{16, 17};
    aclDestroyTensorDesc(inputDesc[0]);
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind3, ACL_FORMAT_ND);
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclopDestroyAttr(opAttr);
}

TEST(UTEST_ACL_Resource_Manager, MatchModelSetCompileFlagTest)
{
    SetCastHasTruncateAttr(false);
    OpModelDef modelDef;
    modelDef.opType = "Cast";
    int64_t shape[]{16, -1};
    int64_t shapeStatic[]{16, 16};
    int64_t range[2][2] = {{16, 16}, {1, 16}};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef.inputDescArr[0], 2, range);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");
    modelDef.opAttr.SetAttr<string>("truncate", "1");

    auto &instance = AclOpResourceManager::GetInstance();
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, true, isDeduplicate), ACL_SUCCESS);

    AclOp aclOp;
    aclopAttr *opAttr = aclopCreateAttr();
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    int64_t shapeFind[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);

    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;

    OpModel opModel;
    bool isDynamic;
    aclOp.opType = "Cast";
    aclOp.numInputs = 1;
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    aclOp.opAttr = opAttr;
    aclopSetAttrString(opAttr, "testAttr", "invalid");
    const_cast<aclTensorDesc *>(aclOp.outputDesc[0])->memtype = ACL_MEMTYPE_HOST;
    const_cast<aclTensorDesc *>(aclOp.outputDesc[0])->isConst = false;
    aclDataBuffer *outputs[1] = {nullptr};
    char ptr[4] = {0};
    outputs[0] = aclCreateDataBuffer(ptr, 4);
    aclOp.outputs = outputs;
    const_cast<aclTensorDesc *>(aclOp.outputDesc[0])->memtype = ACL_MEMTYPE_DEVICE;
    const_cast<aclTensorDesc *>(aclOp.outputDesc[0])->isConst = false;
    aclOp.outputs = nullptr;
    aclDestroyDataBuffer(outputs[0]);
    aclopSetAttrString(opAttr, "testAttr", "attrValue");
    aclOp.isCompile = false;
    SetGlobalCompileFlag(0);
    SetGlobalJitCompileFlag(1);
    EXPECT_EQ(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_ERROR_OP_NOT_FOUND);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclopDestroyAttr(opAttr);
}

TEST(UTEST_ACL_Resource_Manager, MatchModelDynamicHashTest)
{
    SetCastHasTruncateAttr(false);
    OpModelDef modelDef;
    modelDef.opType = "Cast";
    int64_t shape[]{16, -1};
    int64_t shapeStatic[]{16, 16};
    int64_t range[2][2] = {{16, 16}, {1, 16}};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef.inputDescArr[0], 2, range);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");

    auto &instance = AclOpResourceManager::GetInstance();
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, true, isDeduplicate), ACL_SUCCESS);

    AclOp aclOp;
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    int64_t shapeFind[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);
    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;

    OpModel opModel;
    bool isDynamic;
    aclOp.opType = "Cast";
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    aclopAttr *opAttr = aclopCreateAttr();
    aclopSetAttrString(opAttr, "testAttr", "attrValue");
    aclOp.opAttr = opAttr;

    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);

    OpModelDef modelDef_2;
    modelDef.opType = "test_acl";
    int64_t shape_2[]{16, -1};
    int64_t shapeStatic_2[]{16, 16};
    int64_t range_2[2][2] = {{16, 16}, {1, 16}};
    modelDef_2.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape_2, ACL_FORMAT_ND);
    modelDef_2.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic_2, ACL_FORMAT_ND);
    modelDef_2.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic_2, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef_2.inputDescArr[0], 2, range_2);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");

    auto modelDefPtr = shared_ptr<OpModelDef>(new (std::nothrow)OpModelDef(std::move(modelDef_2)));
    size_t k = 17836075261947842321ULL;
    instance.opModels_.hashMap_[k].push_back(std::move(modelDefPtr));

    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);

    instance.opModels_.hashMap_[k].clear();
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclopDestroyAttr(aclOp.opAttr);
}

TEST(UTEST_ACL_Resource_Manager, MatchModelHashTest)
{
    OpModelDef modelDef;
    modelDef.opType = "testOp";
    int64_t shape[]{16, 16};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");

    auto &instance = AclOpResourceManager::GetInstance();
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, false, isDeduplicate), ACL_SUCCESS);

    AclOp aclOp;
    aclopAttr *opAttr = aclopCreateAttr();
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];

    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;

    OpModel opModel;
    bool isDynamic;
    aclOp.opType = "testOp";
    aclopSetAttrString(opAttr, "testAttr", "attrValue");
    aclOp.opAttr = opAttr;
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);

    OpModelDef modelDef_2;
    modelDef.opType = "acltest";
    int64_t shape_2[]{16, 16};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape_2, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape_2, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shape_2, ACL_FORMAT_ND);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");

    auto modelDefPtr = shared_ptr<OpModelDef>(new (std::nothrow)OpModelDef(std::move(modelDef_2)));
    instance.opModels_.hashMap_[6687538955415257199].push_back(std::move(modelDefPtr));

    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclopDestroyAttr(opAttr);

    instance.opModels_.hashMap_[6687538955415257199].clear();
}

TEST(UTEST_ACL_Resource_Manager, MatchModelTest)
{
    OpModelDef modelDef;
    modelDef.opType = "testOp";
    int64_t shape[]{16, 16};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");

    auto &instance = AclOpResourceManager::GetInstance();
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, false, isDeduplicate), ACL_SUCCESS);

    OpModelDef modelDef_2;
    modelDef.opType = "acltest";
    int64_t shape_2[]{16, 16};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape_2, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape_2, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shape_2, ACL_FORMAT_ND);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");

    AclOp aclOp;
    aclopAttr *opAttr = aclopCreateAttr();
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;

    OpModel opModel;
    bool isDynamic;
    aclOp.opType = "testOp";
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.numInputs = 1;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.numInputs = 2;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.numOutputs = 1;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclOp.opAttr = opAttr;
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclopSetAttrString(opAttr, "testAttr", "invalid");
    EXPECT_NE(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_SUCCESS);
    aclopSetAttrString(opAttr, "testAttr", "attrValue");
    EXPECT_EQ(instance.MatchOpModel(aclOp, opModel, isDynamic), ACL_ERROR_FAILURE);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclopDestroyAttr(opAttr);
}

TEST(UTEST_ACL_Resource_Manager, TestStaticMapAging)
{
    OpModelDef modelDef;
    modelDef.opType = "testOpStatic";
    int64_t shape[]{16, 16};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");

    auto &instance = AclOpResourceManager::GetInstance();
    EXPECT_EQ(instance.HandleMaxOpQueueConfig("{\"max_opqueue_num\" : \"1\"}"), ACL_SUCCESS);
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, false, isDeduplicate, false), ACL_SUCCESS);

    OpModelDef modelDef1;
    modelDef1.opType = "testOpStatic1";
    modelDef1.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef1.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef1.outputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef1.opAttr.SetAttr<string>("testAttr", "attrValue");
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef1), instance.opModels_, false, isDeduplicate, false), ACL_SUCCESS);
}

TEST(UTEST_ACL_Resource_Manager, TestDynamicMapAging)
{
    OpModelDef modelDef;
    modelDef.opType = "testOpDynamic";
    int64_t shape[]{16, -1};
    int64_t shapeStatic[]{16, 16};
    int64_t range[2][2] = {{16, 16}, {1, 16}};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef.inputDescArr[0], 2, range);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");

    auto &instance = AclOpResourceManager::GetInstance();
    ASSERT_EQ(instance.HandleMaxOpQueueConfig("{\"max_opqueue_num\" : \"1\"}"), ACL_SUCCESS);
    bool isDeduplicate = false;
    ASSERT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, true, isDeduplicate, false), ACL_SUCCESS);

    OpModelDef modelDef1;
    modelDef1.opType = "testOpDynamic1";
    modelDef1.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef1.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    modelDef1.outputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef1.inputDescArr[0], 2, range);
    modelDef1.opAttr.SetAttr<string>("testAttr", "attrValue");
    ASSERT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, true, isDeduplicate, false), ACL_SUCCESS);
}

TEST(UTEST_ACL_Resource_Manager, TestGetOpModel)
{
    AclOp aclOp;
    aclOp.opType = "newOp";
    auto &instance = AclOpResourceManager::GetInstance();
    EXPECT_NE(instance.GetOpModel(aclOp), ACL_SUCCESS);

}

TEST(UTEST_ACL_Resource_Manager, TestOmFileFilterFn)
{
    ASSERT_TRUE(AclOpResourceManager::OmFileFilterFn("a.om"));
    ASSERT_TRUE(AclOpResourceManager::OmFileFilterFn("aaa.om"));
    ASSERT_TRUE(AclOpResourceManager::OmFileFilterFn("a_123.om"));
    ASSERT_TRUE(AclOpResourceManager::OmFileFilterFn(".om"));
    ASSERT_FALSE(AclOpResourceManager::OmFileFilterFn("a_123.o"));
    ASSERT_FALSE(AclOpResourceManager::OmFileFilterFn("a_123.m"));
    ASSERT_FALSE(AclOpResourceManager::OmFileFilterFn("a_123o.m"));
    ASSERT_FALSE(AclOpResourceManager::OmFileFilterFn("a_123om"));
    ASSERT_FALSE(AclOpResourceManager::OmFileFilterFn("om"));
}

TEST(UTEST_ACL_Resource_Manager, DebugString)
{
    aclTensorDesc desc;
    desc.isConst = true;
    auto *data = new (std::nothrow) int[4];
    std::shared_ptr<void> modelData;
    modelData.reset(data, [](const int *p) { delete[]p; });
    desc.constDataBuf = modelData;
    desc.constDataLen = 4;
    std::string str = desc.DebugString();
    ASSERT_FALSE(str.empty());
    vector<int64_t> shape{1};
    desc.UpdateTensorShape(shape);
    std::pair<int64_t, int64_t> range;
    EXPECT_EQ(desc.dims.size(), shape.size());
    std::vector<std::pair<int64_t, int64_t>> ranges{range};
    desc.UpdateTensorShapeRange(ranges);
    EXPECT_EQ(desc.shapeRange.size(), ranges.size());
}

TEST(UTEST_ACL_Resource_Manager, HandleMaxOpQueueConfigTest)
{
    AclOpResourceManager modelManager;
    aclError ret = modelManager.HandleMaxOpQueueConfig("{\"max_opqueue_num\" : \"0\"}");
    EXPECT_NE(ret, ACL_SUCCESS);

    ret = modelManager.HandleMaxOpQueueConfig("{\"max_opqueue_num\" : \"0\"}");
    EXPECT_NE(ret, ACL_SUCCESS);
}

TEST(UTEST_ACL_Resource_Manager, LoadModelFromMemTest)
{
    size_t modelSize = 20;
    auto *aclModelData = new (std::nothrow) char[modelSize];
    bool isStatic = false;
    auto &instance = AclOpResourceManager::GetInstance();
    EXPECT_NE(instance.LoadModelFromMem(aclModelData, modelSize, isStatic), ACL_SUCCESS);
    delete []aclModelData;
}

TEST(UTEST_ACL_Resource_Manager, BuildOpModelTest)
{
    AclOp aclop;
    auto &instance = AclOpResourceManager::GetInstance();
    EXPECT_NE(instance.BuildOpModel(aclop), ACL_SUCCESS);
}

TEST(UTEST_ACL_Resource_Manager, BuildOpModelTestSuccess)
{
    AclOp aclop;
    std::map<std::string, std::string> options;
    OpCompileService service;
    service.RegisterCreator(NATIVE_COMPILER, &LocalCompiler::CreateCompiler);
    std::shared_ptr<void> modelData;
    size_t modelSize;
    OpCompileService::GetInstance().SetCompileStrategy(NATIVE_COMPILER, options);
    EXPECT_EQ(OpCompileService::GetInstance().CompileOp(aclop, modelData, modelSize), ACL_SUCCESS);
}

ge::Status BuildSingleOpModel_Invoke(ge::OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                   const std::vector<GeTensor> &outputs, ge::OpEngineType engine_type,
                                   int32_t compile_flag, ge::ModelBufferData &model_buff)
{
    (void) op_desc;
    (void) inputs;
    (void) outputs;
    (void) engine_type;
    (void) compile_flag;
    model_buff.length = sizeof(ge::ModelFileHeader) + 1024UL;
    std::unique_ptr<uint8_t[]> buff(new uint8_t[model_buff.length]);
    model_buff.data.reset(reinterpret_cast<uint8_t *>(buff.release()), std::default_delete<uint8_t[]>());
    auto *const file_header = reinterpret_cast<ge::ModelFileHeader *>(model_buff.data.get());
    file_header->model_length = 1024UL;
    return ge::SUCCESS;
}

TEST(UTEST_ACL_Resource_Manager, GetOpModelSucc)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), BuildSingleOpModel(_,_,_,_,_,_))
        .WillOnce(Invoke(BuildSingleOpModel_Invoke));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListTensor(_, _, _))
        .WillRepeatedly(Return(true));

    AclOp aclop;
    std::map<std::string, std::string> options;
    OpCompileService service;
    service.RegisterCreator(NATIVE_COMPILER, &LocalCompiler::CreateCompiler);
    OpCompileService::GetInstance().SetCompileStrategy(NATIVE_COMPILER, options);
    EXPECT_EQ(AclOpResourceManager::GetInstance().GetOpModel(aclop), ACL_SUCCESS);
}

extern bool GetInt_invoke(ge::AttrUtils::ConstAttrHolderAdapter obj, const string &name, int32_t& value);

bool GetBool_true_invoke(ge::AttrUtils::ConstAttrHolderAdapter obj, const string &name, bool &value)
{
  (void) obj;
  (void) name;
  value = true;
  return true;
}

TEST(UTEST_ACL_Resource_Manager, LoadAllModelsTest)
{
    EXPECT_EQ(AclOpResourceManager::GetInstance().LoadAllModels("./"), ACL_SUCCESS);
}

INT32 mmScandir2_invoke(const CHAR *path, mmDirent2 ***entryList, mmFilter2 filterFunc,  mmSort2 sort)
{
    (void) path;
    (void) filterFunc;
    (void) sort;
    static mmDirent2 *dirArray[1];
    static mmDirent2 dirent;
    dirent.d_type = MM_DT_REG;
    strcpy(dirent.d_name, "test.om");
    dirArray[0] = &dirent;
    *entryList = dirArray;
    return 1;
}

bool ReadBytesFromBinaryFileInvoke(char const *file_name, char **buffer, int &length)
{
    (void) file_name;
    const size_t headerSize = sizeof(ge::ModelFileHeader);
    // headerHolder will be released in opModel.data
    char *headerHolder = new (std::nothrow) char[headerSize];
    ge::ModelFileHeader *header = new (headerHolder) ge::ModelFileHeader;
    header->length = 128;
    length = headerSize + header->length;
    *buffer = reinterpret_cast<char *>(header);
    return true;
}

TEST(UTEST_ACL_Resource_Manager, LoadAllModelsFileExistedTest)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ReadBytesFromBinaryFile(_, _, _))
            .WillRepeatedly(Invoke(ReadBytesFromBinaryFileInvoke));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListTensor(_, _, _))
            .WillRepeatedly(Return(true));
    const std::string configDir = ACL_BASE_DIR "/tests/acl_ut/ut/acl/json/";
    EXPECT_EQ(AclOpResourceManager::GetInstance().LoadAllModels(configDir), ACL_SUCCESS);
}

TEST(UTEST_ACL_Resource_Manager, LoadModelFromMemFailedTest)
{

    int modelSize = 10;
    char* model = new(std::nothrow) char[modelSize]();
    auto &instance = AclOpResourceManager::GetInstance();
    EXPECT_NE(instance.LoadModelFromMem(model, modelSize), ACL_SUCCESS);

    modelSize = 257;
    auto *aclModelData = new (std::nothrow) char[modelSize];
    bool isStatic = false;
    EXPECT_NE(instance.LoadModelFromMem(aclModelData, modelSize, isStatic), ACL_SUCCESS);
    delete []aclModelData;
    delete []model;
}

TEST(UTEST_ACL_Resource_Manager, LoadModelFromSharedMemTest)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Init(_, _))
            .WillRepeatedly(Return(0));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelPartition(_, _))
            .WillRepeatedly(Return(0));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Load(_, _, _))
            .WillRepeatedly(Return(0));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListTensor(_, _, _))
            .WillRepeatedly(Return(true));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetInt(_, _, _))
            .WillRepeatedly(Invoke(GetInt_invoke));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), HasAttr(_, _))
            .WillRepeatedly(Return(true));
    size_t modelSize = sizeof(struct ge::ModelFileHeader) + 128UL;
    auto &instance = AclOpResourceManager::GetInstance();
    ge::ModelFileHeader header;
    header.length = 128;
    std::shared_ptr<void> modelData;
    modelData.reset(&header, [](void *) {});
    instance.opModels_.maxOpNum = 100;
    size_t cache_num = instance.modelCache_.cachedModels_.size();
    ++cache_num;
    EXPECT_EQ(instance.LoadModelFromSharedMem(modelData, modelSize, nullptr, false), ACL_SUCCESS);
    EXPECT_EQ(cache_num, instance.modelCache_.cachedModels_.size());
    EXPECT_EQ(instance.LoadModelFromSharedMem(modelData, modelSize, nullptr, false), ACL_SUCCESS);
    EXPECT_EQ(cache_num, instance.modelCache_.cachedModels_.size());
}

TEST(UTEST_ACL_Resource_Manager, LoadModelFromSharedMemDynamicModelTest)
{
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), Init(_, _))
          .WillRepeatedly(Return(0));

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelPartition(_, _))
          .WillRepeatedly(Return(0));

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), Load(_, _, _))
          .WillRepeatedly(Return(0));

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListTensor(_, _, _))
          .WillRepeatedly(Return(true));

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetInt(_, _, _))
          .WillRepeatedly(Return(false));

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), HasAttr(_, _))
          .WillRepeatedly(Return(true));
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetBool(_, _, _))
          .WillRepeatedly(Invoke(GetBool_true_invoke));
  size_t modelSize = sizeof(struct ge::ModelFileHeader) + 128UL;
  auto &instance = AclOpResourceManager::GetInstance();
  ge::ModelFileHeader header;
  header.length = 128;
  std::shared_ptr<void> modelData;
  modelData.reset(&header, [](void *) {});
  ShapeRangeUtils::GetInstance().opModelRanges_.clear();
  size_t rangeInfo = ShapeRangeUtils::GetInstance().opModelRanges_.size();
  EXPECT_EQ(instance.LoadModelFromSharedMem(modelData, modelSize, nullptr, false), ACL_SUCCESS);
  EXPECT_EQ(rangeInfo + 1UL, ShapeRangeUtils::GetInstance().opModelRanges_.size());
}

TEST(UTEST_ACL_Resource_Manager, ModelHashCheckTest)
{
    OpModelDef modelDef;
    modelDef.opType = "acltest";
    int64_t shape[]{16, 16};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");
    auto modelDefPtr = shared_ptr<OpModelDef>(new (std::nothrow)OpModelDef(std::move(modelDef)));

    AclOp aclOp;
    aclOp.opType = "acltesterror";
    aclopAttr *opAttr = nullptr;
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatch(aclOp, opAttr, modelDefPtr), false);

    aclOp.opType = "acltest";
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatch(aclOp, opAttr, modelDefPtr), false);

    aclOp.numInputs = 1;
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatch(aclOp, opAttr, modelDefPtr), false);

    aclOp.numInputs = 2;
    const aclTensorDesc *inputDesc[2];
    int64_t shape2[]{1, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape2, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape2, ACL_FORMAT_ND);
    aclOp.inputDesc = inputDesc;
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatch(aclOp, opAttr, modelDefPtr), false);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);

    aclOp.numInputs = 2;
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclOp.inputDesc = inputDesc;

    aclOp.numOutputs = 2;
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatch(aclOp, opAttr, modelDefPtr), false);

    aclOp.numOutputs = 1;
    const aclTensorDesc *outputDesc[1];
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape2, ACL_FORMAT_ND);
    aclOp.outputDesc = outputDesc;
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatch(aclOp, opAttr, modelDefPtr), false);
    aclDestroyTensorDesc(outputDesc[0]);

    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclOp.outputDesc = outputDesc;
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatch(aclOp, opAttr, modelDefPtr), false);

    aclopSetAttrString(opAttr, "testAttr", "attrValue");
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatch(aclOp, opAttr, modelDefPtr), false);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclopDestroyAttr(opAttr);
}

TEST(UTEST_ACL_Resource_Manager, ValueRangeCheckTest)
{
    std::map<AttrRangeType, ge::GeAttrValue> valueRange;
    void *data = malloc(4);
    aclDataBuffer *dataBuffer = aclCreateDataBuffer(data, 4);
    aclDataType dataType = ACL_FLOAT;
    auto ret = ValueRangeCheck(valueRange, dataBuffer, dataType);
    EXPECT_EQ(ret ,true);

    dataType = ACL_FLOAT16;
    ret = ValueRangeCheck(valueRange, dataBuffer, dataType);
    EXPECT_EQ(ret ,true);

    dataType = ACL_INT8;
    ret = ValueRangeCheck(valueRange, dataBuffer, dataType);
    EXPECT_EQ(ret ,true);

    dataType = ACL_UINT8;
    ret = ValueRangeCheck(valueRange, dataBuffer, dataType);
    EXPECT_EQ(ret ,true);

    dataType = ACL_INT16;
    ret = ValueRangeCheck(valueRange, dataBuffer, dataType);
    EXPECT_EQ(ret ,true);

    free(data);
    aclDestroyDataBuffer(dataBuffer);
}

TEST(UTEST_ACL_Resource_Manager, CheckSHapeRangeTest)
{
    AclOp aclOp;
    const aclTensorDesc *inputDesc[1];
    int64_t shapeFind[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shapeFind, ACL_FORMAT_ND);
    aclOp.inputDesc = inputDesc;

    aclOp.opType = "Cast";
    aclOp.numInputs = 1;

    OpRangeInfo rangeInfo;
    bool ret = ShapeRangeUtils::GetInstance().CheckShapeRange(aclOp, rangeInfo);
    EXPECT_NE(ret, true);
    aclDestroyTensorDesc(inputDesc[0]);
}
namespace {
struct AllocatorStub : public ge::Allocator {
  ge::MemBlock *Malloc(size_t size) override {
    (void) size;
    return nullptr;
  }

  void Free(ge::MemBlock *block) override {
    (void) block;
  }
  ~AllocatorStub() {}
};
std::unique_ptr<ge::Allocator> CreateAllocators_succ(const gert::TensorPlacement &placement)
{
  (void) placement;
  AllocatorStub *ptr = new (std::nothrow) AllocatorStub();
  return std::unique_ptr<ge::Allocator>(ptr);
}

std::unique_ptr<ge::Allocator> CreateAllocators_fail(const gert::TensorPlacement &placement)
{
  (void) placement;
  return nullptr;
}
} // namespace

TEST(UTEST_ACL_Resource_Manager, TestStream) {
    AclResourceManager &instance = AclResourceManager::GetInstance();
    EXPECT_NE(&instance, nullptr);

    const aclrtStream stream1 = (void *)0x2022;
    const aclrtStream stream2 = (void *)0x2023;
    instance.CleanAllocators(stream1);
    instance.CleanAllocators(stream2);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
            .WillRepeatedly(Invoke(CreateAllocators_fail));

    // first call GetAllocators, CreateDefaultAllocator fail
    gert::Allocators *ptr = instance.GetAllocators(stream1).get();
    EXPECT_EQ(ptr, nullptr);

    // second call GetExternalAllocators, CreateDefaultAllocator succ
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
            .WillRepeatedly(Invoke(CreateAllocators_succ));
    ptr = instance.GetAllocators(stream1).get();
    // 修改后，工厂类调用create allocator失败，则终止流程，因此，下次会重新创建
    EXPECT_NE(ptr, nullptr);

    gert::Allocators *allocatorPtr = ptr;
    // first call GetAllocators, CreateDefaultAllocator succ
    ptr = instance.GetAllocators(stream2).get();
    EXPECT_NE(ptr, allocatorPtr);

    // second call GetExternalAllocators, CreateDefaultAllocator succ
    allocatorPtr = ptr;
    ptr = instance.GetAllocators(stream2).get();
    EXPECT_EQ(ptr, allocatorPtr); // old value

    // clean all stream
    instance.CleanAllocators(stream1);
    instance.CleanAllocators(stream2);
    // last prt for stream1 and stream2 is not nullptr
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
            .WillRepeatedly(Invoke(CreateAllocators_fail));
    ptr = instance.GetAllocators(stream1).get();
    EXPECT_EQ(ptr, nullptr); // new value
    ptr = instance.GetAllocators(stream2).get();
    EXPECT_EQ(ptr, nullptr); // new value

    instance.CleanAllocators(stream1);
    instance.CleanAllocators(stream2);
}

TEST(UTEST_ACL_Resource_Manager, TestDestroyStream)
{
    AclResourceManager &instance = AclResourceManager::GetInstance();
    EXPECT_NE(&instance, nullptr);

    const aclrtStream stream1 = (void *)0x2022;
    instance.CleanAllocators(stream1);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtCtxGetCurrentDefaultStream(_))
            .WillRepeatedly(Return(ACL_RT_SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
            .WillRepeatedly(Invoke(CreateAllocators_succ));

    gert::Allocators *ptr = instance.GetAllocators(stream1, true).get();
    EXPECT_NE(ptr, nullptr);

    aclrtDestroyStream(stream1);
    instance.HandleReleaseSourceByStream(stream1, ACL_RT_STREAM_STATE_DESTROY_PRE, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtAllocatorGetByStream(_,_,_,_,_,_,_))
            .WillRepeatedly(Return(ACL_ERROR_FAILURE));
    ptr = instance.GetAllocators(stream1, false).get();
    EXPECT_EQ(ptr, nullptr);
    instance.CleanAllocators(stream1);
}

TEST(UTEST_ACL_Resource_Manager, TestResetDevice)
{
    AclResourceManager &instance = AclResourceManager::GetInstance();
    EXPECT_NE(&instance, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
            .WillRepeatedly(Invoke(CreateAllocators_succ));

    const aclrtStream stream = nullptr;

    g_priCtx = (void *)0xbb;
    instance.CleanAllocators(g_priCtx);
    void *ptr = instance.GetAllocators(stream).get();
    EXPECT_EQ(ptr, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
            .WillRepeatedly(Invoke(CreateAllocators_fail));
    ptr = instance.GetAllocators(stream).get();
    EXPECT_EQ(ptr, nullptr);
    instance.CleanAllocators(g_priCtx);
}

TEST(UTEST_ACL_Resource_Manager, GetExternalAllocatorsSuccess)
{
    AclResourceManager &instance = AclResourceManager::GetInstance();
    EXPECT_NE(&instance, nullptr);

    aclrtStream stream = (aclrtStream)0x10;
    // The following test code is used to test the logic of allocator update when the
    // internal allocator and external allocator are used in different orders.
    // The "old_desc_exist" indicates the allocator description cached by GE in the
    // past, while "new_desc_exist" refers to the latest externally registered allocator
    // description.
    // !old_desc_exist && new_desc_exist.
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtAllocatorGetByStream(_,_,_,_,_,_,_))
            .WillRepeatedly(Return(ACL_SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
            .WillRepeatedly(Invoke(CreateAllocators_succ));
    gert::Allocators *allocatorPtr = instance.GetAllocators(stream).get();
    EXPECT_NE(allocatorPtr, nullptr);
    // old_desc_exist && new_desc_exist.
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtAllocatorGetByStream(_,_,_,_,_,_,_))
            .WillRepeatedly(Return(ACL_SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
            .WillRepeatedly(Invoke(CreateAllocators_succ));
    gert::Allocators *allocatorPtr1 = instance.GetAllocators(stream).get();
    EXPECT_NE(allocatorPtr1, nullptr);
    // old_desc_exist && !new_desc_exist
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtAllocatorGetByStream(_,_,_,_,_,_,_))
            .WillRepeatedly(Return(ACL_ERROR_FAILURE));
    gert::Allocators *allocatorPtr2 = instance.GetAllocators(stream).get();
    EXPECT_NE(allocatorPtr2, nullptr);
}

TEST(UTEST_ACL_Resource_Manager, RegisterModelTest_ShaperangeIndexOutDimIndex)
{
    OpModelDef modelDef;
    modelDef.opType = "testOp";
    int64_t shape[]{16, -1};
    int64_t shape_generate[]{-2};
    int64_t shapeStatic[]{16, 16};
    int64_t range[2][2] = {{16, 16}, {1, 16}};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shapeStatic, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 1, shape_generate, ACL_FORMAT_ND);
    aclSetTensorShapeRange(&modelDef.inputDescArr[0], 2, range);
    aclSetTensorShapeRange(&modelDef.outputDescArr[0], 2, range);

    auto &instance = AclOpResourceManager::GetInstance();
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), instance.opModels_, true, isDeduplicate, false), ACL_SUCCESS);
}

TEST(UTEST_ACL_Resource_Manager, RegisterModelTest_SameCache)
{
    auto &instance = AclOpResourceManager::GetInstance();
    instance.opModels_.maxOpNum = 1000;
    AclOp aclOp;
    aclOp.opType = "test_static";
    OpModelDef modelConfig;
    modelConfig.opModelId = 999;
    modelConfig.opType = "test_static";
    auto modelDefPtr = std::make_shared<OpModelDef>(modelConfig);
    std::shared_ptr<OpModelDef> agingModelDef = nullptr;
    bool isRegistered = false;
    auto ret = instance.opModels_.Insert(aclOp, modelDefPtr, agingModelDef, isRegistered);
    EXPECT_EQ(ret ,ACL_SUCCESS);
    EXPECT_FALSE(isRegistered);

    modelConfig.opModelId = 1000;
    auto modelDefPtr2 = std::make_shared<OpModelDef>(modelConfig);
    ret = instance.opModels_.Insert(aclOp, modelDefPtr2, agingModelDef, isRegistered);
    EXPECT_EQ(ret ,ACL_SUCCESS);
    EXPECT_TRUE(isRegistered);

    std::shared_ptr<OpModelDef> acquireModelDef = nullptr;
    instance.opModels_.Get(aclOp, acquireModelDef, false);
    EXPECT_NE(acquireModelDef, nullptr);
    EXPECT_EQ(acquireModelDef->opModelId, 999);
}

TEST(UTEST_ACL_Resource_Manager, RegisterModelTest_SameDynamicCache)
{
    auto &instance = AclOpResourceManager::GetInstance();
    instance.opModels_.maxOpNum = 1000;
    AclOp aclOp;
    aclOp.opType = "test_dynamic";
    OpModelDef modelConfig;
    modelConfig.opModelId = 9999;
    modelConfig.opType = "test_dynamic";
    modelConfig.seq = 888;
    auto modelDefPtr = std::make_shared<OpModelDef>(modelConfig);
    std::shared_ptr<OpModelDef> agingModelDef = nullptr;
    bool isRegistered = false;
    auto ret = instance.opModels_.InsertDynamic(aclOp, modelDefPtr, agingModelDef, isRegistered);
    EXPECT_EQ(ret ,ACL_SUCCESS);
    EXPECT_FALSE(isRegistered);

    modelConfig.opModelId = 10000;
    auto modelDefPtr2 = std::make_shared<OpModelDef>(modelConfig);
    ret = instance.opModels_.InsertDynamic(aclOp, modelDefPtr2, agingModelDef, isRegistered);
    EXPECT_EQ(ret ,ACL_SUCCESS);
    EXPECT_TRUE(isRegistered);

    std::shared_ptr<OpModelDef> acquireModelDef = nullptr;
    instance.opModels_.GetDynamic(aclOp, acquireModelDef, modelConfig.seq, false);
    EXPECT_NE(acquireModelDef, nullptr);
    EXPECT_EQ(acquireModelDef->opModelId, 9999);
}

TEST(UTEST_ACL_Resource_Manager, GetAllocators_Fail_HostAllocatorIsNull)
{
    auto &instance = AclResourceManager::GetInstance();
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_))
        .WillOnce(Invoke(CreateAllocators_succ))
        .WillRepeatedly(Invoke(CreateAllocators_fail));
    aclrtStream stream = (aclrtStream)0x2233;
    auto allocators = instance.GetAllocators(stream, true);
    EXPECT_EQ(allocators, nullptr);
}

TEST(UTEST_ACL_Resource_Manager, GetAllocators_Fail_DeviceAllocatorIsNull)
{
    auto &instance = AclResourceManager::GetInstance();
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtAllocatorGetByStream(_,_,_,_,_,_,_))
        .WillOnce(Return(false));
    aclrtStream stream = (aclrtStream)0x2233;
    auto allocators = instance.GetAllocators(stream, true).get();
    EXPECT_EQ(allocators, nullptr);
}

TEST(UTEST_ACL_Resource_Manager, GetKeyByStreamOrDefaultStreamTest)
{
    auto &instance = AclResourceManager::GetInstance();
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtCtxGetCurrentDefaultStream(_))
        .WillOnce(Return(ACL_ERROR_RT_FAILURE));
    aclrtStream stream = nullptr;
    auto retStream = instance.GetKeyByStreamOrDefaultStream(stream);
    EXPECT_EQ(retStream, nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtCtxGetCurrentDefaultStream(_))
        .WillRepeatedly(Return(ACL_RT_SUCCESS));
    stream = (aclrtStream)0x2233;
    retStream = instance.GetKeyByStreamOrDefaultStream(stream);
    EXPECT_NE(retStream, nullptr);
}
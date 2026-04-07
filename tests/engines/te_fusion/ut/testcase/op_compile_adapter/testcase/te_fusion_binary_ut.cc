/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>
#include <gtest/gtest_pred_impl.h>
#include "../stub/Python_stub.h"
#include "common_stub.h"
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>

#define private public
#define protected public

#include "graph/node.h"
#include "graph/op_desc.h"
#include "tensor_engine/fusion_api.h"
#include "compile/fusion_manager.h"
#include "register/op_check.h"
#include "common/common_utils.h"
#include "common/fusion_common.h"
#include "common/tbe_op_info_cache.h"
#include "common/te_config_info.h"
#include "common/te_file_utils.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/ascend_string.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/operator.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "te_fusion_base.h"
#include "cache/te_cache_space_manager.h"
#include "binary/generate_simple_key.h"
#include "binary/fusion_binary_info.h"
#include "binary/binary_manager.h"
#include "cache/te_cache_manager.h"
// #include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "graph/ge_local_context.h"
#include "binary/shape_generalization.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace te;
using namespace te::fusion;
using TbeOpInfoPtr = std::shared_ptr<te::TbeOpInfo>;

class TeFusionBinaryUTest : public testing::Test
{
    public:
        TeFusionBinaryUTest(){}
    protected:
        virtual void SetUp()
        {
        }
        virtual void TearDown()
        {
        }
    protected:

};

void CreateOfile(std::string &fileName) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    std::string filePath = te::fusion::RealPath(currentFilePath) + fileName;
    FILE* fp = fopen(filePath.c_str(), "w+");
    if (fp == nullptr) {
        printf("Open file[%s] failed. \r\n", filePath.c_str());
        return;
    }
    int res = chmod(filePath.c_str(), 0640);
    if (res == -1) {
        printf("Update file[%s] authority failed. \r\n", filePath.c_str());
        fclose(fp);
        return;
    }
    fclose(fp);
}

bool GetOptionInfo_stub(te::fusion::TeFusionManager *This, bool isThreadLocal, const std::string &key, std::string &value)
{
    if (key == "ge.jit_compile") {
        value = "2";
    }

    return true;
}

bool testFunc(const ge::Operator &op, ge::AscendString &result) {
    return true;
}

UINT32 GenSimplifiedKeyKernelFuncStub(gert::TilingContext *tiling_context, ge::char_t *res) {
  strcat_s(res, 255, "p=[null,null]/float");
  return 0;
}

TEST(TeFusionBinaryUTest, createOfile) {
    printf("createOfile. \r\n");
    // create te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.o
    std::string ofileName = "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.o";
    CreateOfile(ofileName);
    // create te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.om
    ofileName = "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.om";
    CreateOfile(ofileName);
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_path_version) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    setenv("ASCEND_OPP_PATH", currentFilePath.c_str(), 1);
    TeConfigInfo::Instance().InitEnvItem();
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "ascend910";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::ShortSocVersion)] = "ascend910";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::CoreType)] = "ascend910";
    const std::map<std::string, std::string> options = {
        {"op.binary.builtin", "6|/op_impl/built-in/ai_core/tbe/kernel/"},
        {"op.binary.custom", "2|/op_impl/custom/ai_core/tbe/kernel/"},
        {"om.binary.builtin", "6|/op_impl/built-in/ai_core/tbe/model/"},
        {"om.binary.custom", "2|/op_impl/custom/ai_core/tbe/model/"}
    };
    // get binary file path
    te::fusion::BinaryManager::Instance().GetBinaryOppPath(options);
    te::fusion::BinaryManager::Instance().GetAllBinaryVersionInfo(true);
    te::fusion::BinaryManager::Instance().GetAllBinaryVersionInfo(false);
    // check soc path
    std::string oppPath = te::fusion::RealPath(currentFilePath) + "/op_impl/built-in/ai_core/tbe/kernel/config/ascend910/";
    std::string customOppPath = te::fusion::RealPath(currentFilePath) + "/op_impl/custom/ai_core/tbe/kernel/config/";
    std::string oppKernelPath = te::fusion::RealPath(currentFilePath) + "/op_impl/built-in/ai_core/tbe/kernel/";
    std::string customOppKernelPath = te::fusion::RealPath(currentFilePath) + "/op_impl/custom/ai_core/tbe/kernel/";
    std::string oppPath1, customOppPath1, oppKernelPath1, customOppKernelPath1;
    auto iter = te::fusion::BinaryManager::Instance().binaryOppPath_.find(6);
    if (iter != te::fusion::BinaryManager::Instance().binaryOppPath_.end()) {
        oppPath1 = iter->second;
    }
    iter = te::fusion::BinaryManager::Instance().binaryOppPath_.find(2);
    if (iter != te::fusion::BinaryManager::Instance().binaryOppPath_.end()) {
        customOppPath1 = iter->second;
    }
    iter = te::fusion::BinaryManager::Instance().binaryOppKernelPath_.find(6);
    if (iter != te::fusion::BinaryManager::Instance().binaryOppKernelPath_.end()) {
        oppKernelPath1 = iter->second;
    }
    iter = te::fusion::BinaryManager::Instance().binaryOppKernelPath_.find(2);
    if (iter != te::fusion::BinaryManager::Instance().binaryOppKernelPath_.end()) {
        customOppKernelPath1 = iter->second;
    }
    EXPECT_EQ(oppPath1, oppPath);
    EXPECT_EQ(customOppPath1, customOppPath);
    EXPECT_EQ(oppKernelPath1, oppKernelPath);
    EXPECT_EQ(customOppKernelPath1, customOppKernelPath);

    // get binary file version and cur version
    std::string kernelPath = te::fusion::RealPath(currentFilePath) + "/";
    std::string binAdkVersion;
    std::string binOppVersion;
    te::fusion::BinaryManager::Instance().GetBinaryVersionInfo(kernelPath, binAdkVersion, binOppVersion);
    std::string binCustomAdkVersion;
    std::string binCustomOppVersion;
    te::fusion::BinaryManager::Instance().GetBinaryVersionInfo(kernelPath, binCustomAdkVersion,
                                                                 binCustomOppVersion);
    // check version
    std::string adkVersion = "1.73.t1.0.b010"; // same as file version.info
    std::string oppVersion = "1.73.t1.0.b010";
    EXPECT_EQ(binAdkVersion, adkVersion);
    EXPECT_EQ(binOppVersion, oppVersion);
    EXPECT_EQ(binCustomAdkVersion, adkVersion);
    EXPECT_EQ(binCustomOppVersion, oppVersion);

    std::string currentFilePath1 = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub";
    std::string kernelPath1 = te::fusion::RealPath(currentFilePath1) + "/";
    std::string binAdkVersion1;
    std::string binOppVersion1;
    te::fusion::BinaryManager::Instance().GetBinaryVersionInfo(kernelPath1, binAdkVersion1,
                                                                 binOppVersion1);
}

static void BinaryTeFusionCreateSingleNodeGraph(ComputeGraphPtr graph, std::vector<ge::Node *> &teGraphNode,
    bool isCustom) {
    OpDescPtr relu_op = std::make_shared<OpDesc>("matmul", "Activation");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");

    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc1);
    data->AddOutputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc1);

    vector<vector<int64_t>> sliceShapeAll; // for SetSgtTensorSliceShape
    vector<int64_t> sliceShape = {1};
    sliceShapeAll.push_back(sliceShape);

    (void)ge::AttrUtils::SetListListInt(relu_op->MutableInputDesc(0), "_sgt_slice_shape", sliceShapeAll);
    (void)ge::AttrUtils::SetListListInt(relu_op->MutableInputDesc(0), "_sgt_ori_slice_shape", sliceShapeAll);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    data->AddInputDesc("z", in_desc3);

    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    ge::AttrUtils::SetInt(relu_node->GetOpDesc(), "_ascendc_sp_sub_id", 1);

    ge::AttrUtils::SetBool(relu_node->GetOpDesc(), "_is_custom_op", isCustom);
    if (isCustom) {
        ge::AttrUtils::SetInt(relu_node->GetOpDesc(), "_fe_imply_type", 2);
    } else {
        ge::AttrUtils::SetInt(relu_node->GetOpDesc(), "_fe_imply_type", 6);
    }
    teGraphNode.push_back(relu_node.get());
}

static void BinaryTeFusionCreateSingleNodeGraph2(ComputeGraphPtr graph, std::vector<ge::Node *> &teGraphNode) {
    OpDescPtr relu_op = std::make_shared<OpDesc>("fill", "Fill");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");

    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc1);
    data->AddOutputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc1);

    vector<vector<int64_t>> sliceShapeAll; // for SetSgtTensorSliceShape
    vector<int64_t> sliceShape = {1};
    sliceShapeAll.push_back(sliceShape);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    data->AddInputDesc("z", in_desc3);

    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    ge::AttrUtils::SetInt(relu_node->GetOpDesc(), "_fe_imply_type", 6);
    teGraphNode.push_back(relu_node.get());
}

static void BinaryTeFusionCreateSingleNodeGraphV2(ComputeGraphPtr graph, std::vector<ge::Node *> &teGraphNode,
    bool isCustom, int switchCase) {
    OpDescPtr relu_op = std::make_shared<OpDesc>("matmul", "Activation");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");

    vector<int64_t> dims;
    if (switchCase == 0) {
        dims = {-1, -1, -1, -1};
    } else if (switchCase == 1) {
        dims = {16, -1, 16, 16};
    }
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc1);
    data->AddOutputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc1);

    vector<vector<int64_t>> sliceShapeAll; // for SetSgtTensorSliceShape
    vector<int64_t> sliceShape = {1};
    sliceShapeAll.push_back(sliceShape);

    (void)ge::AttrUtils::SetListListInt(relu_op->MutableInputDesc(0), "_sgt_slice_shape", sliceShapeAll);
    (void)ge::AttrUtils::SetListListInt(relu_op->MutableInputDesc(0), "_sgt_ori_slice_shape", sliceShapeAll);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    data->AddInputDesc("z", in_desc3);

    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

    ge::AttrUtils::SetBool(relu_node->GetOpDesc(), "_is_custom_op", isCustom);
    if (isCustom) {
        ge::AttrUtils::SetInt(relu_node->GetOpDesc(), "_fe_imply_type", 2);
    } else {
        ge::AttrUtils::SetInt(relu_node->GetOpDesc(), "_fe_imply_type", 6);
    }
    ge::AttrUtils::SetBool(relu_node->GetOpDesc(), "op_jit_compile", false);
    teGraphNode.push_back(relu_node.get());
}

void BinaryAddOpParamToTbeOpInfo(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                           std::vector<std::pair<int64_t, int64_t>> &range, bool isInput, TbeOpInfo &op_info)
{
    TbeOpParam opParam;
    std::vector<TbeOpTensor> tensors;
    TbeOpTensor tensor(name, shape, dtype, format);
    tensor.SetShapeRange(range);
    tensor.SetOriginShape(shape);
    tensor.SetOriginShapeRange(range);
    tensor.SetOriginFormat(format);

    tensors.push_back(tensor);
    opParam.SetTensors(tensors);
    if (name == "input12") {
        opParam.SetValueDepend(VALUE_DEPEND_OPTIONAL);
    }

    if (isInput) {
        op_info.AddInput(opParam);
    } else {
        op_info.AddOutput(opParam);
    }
}

void BinaryAddAttrToTbeOpInfo(TbeOpInfo &opInfo)
{
    TbeAttrValue attr_v_int8("axis", (int8_t)1);

    std::vector<int64_t>  list_int64_v;
    list_int64_v.push_back(1);
    list_int64_v.push_back(2);
    TbeAttrValue attr_v_list_int64("axis1", list_int64_v);

    opInfo.AddAttrValue(attr_v_int8);
    opInfo.AddAttrValue(attr_v_list_int64);
}

bool GeneralizeOps_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_stub \r\n");

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    return true;
}

bool GeneralizeFusionOps_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeFusionOps_stub \r\n");

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_fusionOpParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    return true;
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void GetBinaryPath_stub(te::fusion::BinaryManager *This, const OpBuildTaskPtr &opTask, bool isOm, bool isKernel,
                        std::string &binaryPath)
{
    printf("GetBinaryPath_stub \r\n");
    binaryPath = te::fusion::RealPath(GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub") + "/";
    return;
}

bool GetSimpleKeyMode_stub(te::fusion::BinaryInfoBase *This,
                           const std::string &opType, SimpleKeyModeType &simplekeyMode)
{
    simplekeyMode = SimpleKeyModeType::CUSTOM_MODE;
    return true;
}

TEST(TeFusionBinaryUTest, test_GetBinaryInfoConfigPath) {
    string binaryPath = "./llt/atc/opcompiler/te_fusion/ut/stub/binary_stub/ai_core/tbe/config/";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::ShortSocVersion)] = "ascend910b";
    string configRealPath = ""; 
    BinaryInfoBasePtr binaryInfoPtr = std::make_shared<BinaryInfoBase>();
    binaryInfoPtr->GetBinaryInfoConfigPath(binaryPath, configRealPath, false, "");
}

TEST(TeFusionBinaryUTest, test_SetBinaryConfigPaths) {
   uint64_t implyType = 6;
   string path = "./llt/atc/opcompiler/te_fusion/ut/stub/binary_stub/ai_core/tbe/kernel/";
   BinaryManager::Instance().SetBinaryConfigPaths(implyType, path);
   BinaryManager::Instance().binaryConfigPathMap_ = {{1, ""}};
   BinaryManager::Instance().ParseAllBinaryInfoConfigPath();
}


TEST(TeFusionBinaryUTest, fuzzycompile_binary_reuse_singleop_success) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, true);
    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(2, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(2, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";
    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    std::shared_ptr<TbeOpInfo> pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_stub));

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    EXPECT_EQ(res, OP_BUILD_SUCC);

    res = WaitAllFinished(graphId, tasks); // check task into pInstance->finishedTask_ quene
    EXPECT_EQ(res, false);
    // check binary json file path set ok
    std::string jsonPath;
    (void)AttrUtils::GetStr(task->outNode, "json_file_path", jsonPath);
    std::string jsonFilePath = te::fusion::RealPath(currentFilePath) + "/" +
        "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    EXPECT_EQ(jsonPath.empty(), true);
    GlobalMockObject::verify();
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_reuse_fusionop_success) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    opInfo.SetBuildType(FUZZILY_BUILD);
    BinaryAddAttrToTbeOpInfo(opInfo);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeFusionOps_stub));


    // same as \stub\binary_stub\fusion_ops.json second bin
    (void)ge::AttrUtils::SetStr(teGraphNode[0]->GetOpDesc(), "graph_pattern", "Conv2d_Transdata_2");

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);

    EXPECT_EQ(res, OP_BUILD_SUCC);

    res = WaitAllFinished(graphId, tasks); // check task into pInstance->finishedTask_ quene
    EXPECT_EQ(res, false);
    // check binary json file path set ok
    std::string jsonPath;
    (void)AttrUtils::GetStr(task->outNode, "json_file_path", jsonPath);
    std::string jsonFilePath = te::fusion::RealPath(currentFilePath) + "/" +
        "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    EXPECT_EQ(jsonPath.empty(), true);
    GlobalMockObject::verify();
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_condition_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, true);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(2, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(2, "1.73.t1.0.b011");

    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // version not match
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    BinaryManager::Instance().binOppVersionMap_.emplace(2, "1.73.t1.0.b010");
    // not fuzzy build
    task->buildType = ACCURATELY_BUILD;
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    task->buildType = FUZZILY_BUILD;

    ge::GetThreadLocalContext().global_options_[ge::PERFORMANCE_MODE] = "high";
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    ge::GetThreadLocalContext().global_options_[ge::PERFORMANCE_MODE] = "normal";

    // debug mode
    TeConfigInfo::Instance().config_enum_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigEnumItem::OpDebugLevel)] = 1;
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
}

bool GetJsonValueFromJsonFile_stub(const string &jsonFilePath, nlohmann::json &jsonValue)
{
    printf("GetJsonValueFromJsonFile_stub \r\n");
    return false;
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_check_file_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(2, "1.73.t1.0.b010");
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(2, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");

    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);

    // singleop, huawei op, has filename, json has no binList fail
    opInfo.SetOpFileName("fill_fail1");
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);

    // read file fail
    MOCKER_CPP(&te::fusion::TeFileUtils::GetJsonValueFromJsonFile, bool (*)(const string &, nlohmann::json &))
        .stubs()
        .will(invoke(GetJsonValueFromJsonFile_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // singleop, custom op, has filename, json has no binList fail
    ge::AttrUtils::SetBool(teGraphNode[0]->GetOpDesc(), "_is_custom_op", true);
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);

    // singleop, custom op,  json not exist fail
    opInfo.SetOpFileName("fill_fail3");
    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    std::shared_ptr<TbeOpInfo> pTbeOp1(new (std::nothrow) TbeOpInfo(opInfo));
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp1);
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);

    // fusionop, custom op,  json binList format fail
    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode, false);
    opInfo.SetOpFileName("fill_fail2");
    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    std::shared_ptr<TbeOpInfo> pTbeOp2(new (std::nothrow) TbeOpInfo(opInfo));
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp2);
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_graph_pattern_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    opInfo.SetBuildType(FUZZILY_BUILD);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // node has no "graph_pattern" failed
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);

    // node "graph_pattern" not matched in bin json failed
    (void)ge::AttrUtils::SetStr(teGraphNode[0]->GetOpDesc(), "graph_pattern", "Conv2d_Transdata_5");
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_statickey_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    opInfo.SetOpFileName("fill_nomatch_staticKey");
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);

    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeFusionOps_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();
}

bool GetOpImplModeByOpNode_stub(const ge::Node *opNode,
    std::string &implMode)
{
    printf("GetOpImplModeByOpNode_stub \r\n");
    return false;
}

bool GenerateStrSha256HashValue_stub(const std::string &str, std::string &result)
{
    printf("GenerateStrSha256HashValue_stub \r\n");
    return false;
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_genstatickey_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, true);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strategy;
    json staticKeyJson;
    json binListJson;
    res = te::fusion::BinaryManager::Instance().MatchStaticKey(task, staticKeyJson, binListJson);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();
    res = te::fusion::BinaryManager::Instance().MatchStaticKey(task, staticKeyJson, binListJson);
    EXPECT_EQ(res, false);

    GlobalMockObject::verify();

    task->opNodes[0] = nullptr;
    res = te::fusion::BinaryManager::Instance().MatchStaticKey(task, staticKeyJson, binListJson);
    EXPECT_EQ(res, false);
}

bool GeneralizeOps_Int64mode_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeFusionOps_Int64mode_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["int64Mode"] = true;
    return true;
}

bool GeneralizeOps_range_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeFusionOps_range_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    std::vector<std::pair<int64_t, int64_t>> opRange;
    opRange.emplace_back(std::make_pair(1, 6));
    generalizedResult.dynamicJson["inputs"][0]["range"] = opRange;
    return true;
}

bool GeneralizeOps_oriformat_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeFusionOps_range_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["inputs"][0]["ori_format"] = "NHCW";
    return true;
}

bool GeneralizeOps_orishape_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeFusionOps_range_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    std::vector<int64_t> shape;
    shape.emplace_back(17);
    shape.emplace_back(128);
    generalizedResult.dynamicJson["inputs"][0]["ori_shape"] = shape;
    return true;
}

bool GeneralizeOps_splitindex_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeFusionOps_range_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["outputs"][0]["split_index"] = 44;
    return true;
}

bool GeneralizeOps_l1fusiontype_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeFusionOps_range_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["inputs"][0]["L1_fusion_type"] = 33;
    return true;
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_opparams_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // int64mode fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_Int64mode_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, true);
    GlobalMockObject::verify();

    // range fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_range_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // oriformat��orishape��addr_type��addr_offset... fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_oriformat_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_orishape_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // SPLIT_INDEX
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_splitindex_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // L1_fusion_type
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_l1fusiontype_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();
}

class ReuseTask {
   public:
    ComputeGraphPtr owner_graph;
    shared_ptr<OpBuildTask> task;
    ge::Node* node;

    ReuseTask()
        : owner_graph(std::make_shared<ComputeGraph>("te"))
    {
        uint64_t graphId = 100;
        uint64_t taskId = static_cast<uint64_t>(rand());
        uint64_t sgtThreadIndex = 300;

        std::vector<ge::Node *> teGraphNode;
        BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

        std::string opName = teGraphNode[0]->GetName();
        std::shared_ptr<ge::OpDesc> opDescPtr =
            std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

        auto task =
            std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{
                graphId, taskId, sgtThreadIndex, teGraphNode, opDescPtr,
                te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
        task->buildType = FUZZILY_BUILD;

        TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
        std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
        ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
        pTbeOp->SetNode(nodePtr);
        TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

        this->task = task;
        this->node = teGraphNode[0];
    }
};

void DoTestDeterministicReuse(char* deterministicInBin, char* deterministicInOpParam, bool expectReuse)
{
    ReuseTask task;
    std::string stubPath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub";

    json allBinJson;
    TeFileUtils::GetJsonValueFromJsonFile(stubPath + "/binary_stub/fill.json", allBinJson);

    json binJson = allBinJson["binList"];

    GeneralizedResult generalizedResult;
    TeFileUtils::GetJsonValueFromJsonFile(stubPath + "/binary_stub/fill_opParams.json", generalizedResult.dynamicJson);

    binJson[0]["deterministic"] = string(deterministicInBin);
    generalizedResult.dynamicJson["deterministic"] = string(deterministicInOpParam);

    bool resultIsMatch = BinaryManager::Instance().MatchOpParams(task.task, binJson, generalizedResult);
    ASSERT_EQ(resultIsMatch, expectReuse);
}

TEST(TeFusionBinaryUTest, deterministicReuse_WhenCacheTrue_OptionTrue_WillReuse)
{
    DoTestDeterministicReuse("true", "true", true);
}

TEST(TeFusionBinaryUTest, deterministicReuse_WhenCacheIgnore_OptionTrue_WillReuse)
{
    DoTestDeterministicReuse("ignore", "true", true);
}

TEST(TeFusionBinaryUTest, deterministicReuse_WhenCacheFalse_OptionTrue_WillNotReuse)
{
    DoTestDeterministicReuse("false", "true", false);
}

TEST(TeFusionBinaryUTest, deterministicReuse_WhenCacheTrue_OptionFalse_WillNotReuse)
{
    DoTestDeterministicReuse("true", "false", false);
}

TEST(TeFusionBinaryUTest, deterministicReuse_WhenCacheIgnore_OptionFalse_WillReuse)
{
    DoTestDeterministicReuse("ignore", "false", true);
}

TEST(TeFusionBinaryUTest, deterministicReuse_WhenCacheFalse_OptionFalse_WillReuse)
{
    DoTestDeterministicReuse("false", "false", true);
}

bool GeneralizeOps_attrsnum_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeFusionOps_range_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["attrs"].push_back({});
    return true;
}

bool GeneralizeOps_attrdtype_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_attrdtype_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["attrs"][0]["dtype"] = "string";
    return true;
}

bool GeneralizeOps_uint8attrvalue_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_uint8attrvalue_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["attrs"][3]["value"] = 5;
    return true;
}

bool GeneralizeOps_float32attrvalue_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_float32attrvalue_stub \r\n");

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";

    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["attrs"][8]["value"] = 5.0;
    return true;
}

bool GeneralizeOps_doubleattrvalue_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_doubleattrvalue_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_opParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["attrs"][4]["value"] = 6.0;
    return true;
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_singleopattrs_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // attrs nums not match fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_attrsnum_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // dtype fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_attrdtype_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // uint8 value fail
     MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_uint8attrvalue_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, true);
    GlobalMockObject::verify();

    // float32 value fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_float32attrvalue_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // double value fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps)
        .stubs()
        .will(invoke(GeneralizeOps_doubleattrvalue_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();
}

bool GeneralizeOps_graphopparamsnums_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_graphopparamsnums_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_fusionOpParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    generalizedResult.dynamicJson["graphOpParams"].push_back({});
    return true;
}

bool GeneralizeOps_listuint32attrvalue_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_listuint32attrvalue_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_fusionOpParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    std::vector<int64_t> value;
    value.emplace_back(17);
    value.emplace_back(128);
    generalizedResult.dynamicJson["graphOpParams"][0]["attrs"][6]["value"] = value;
    return true;
}

bool GeneralizeOps_listuint32attrvalue1_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_listuint32attrvalue_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_fusionOpParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    std::vector<int64_t> value;
    generalizedResult.dynamicJson["graphOpParams"][0]["attrs"][6]["value"] = value;
    return true;
}

bool GeneralizeOps_liststringattrvalue_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_listuint32attrvalue_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_fusionOpParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    std::vector<std::string> value;
    value.emplace_back("11");
    value.emplace_back("12");
    value.emplace_back("13");
    generalizedResult.dynamicJson["graphOpParams"][0]["attrs"][10]["value"] = value;
    return true;
}

bool GeneralizeOps_listlistint64attrvalue_stub(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    printf("GeneralizeOps_listlistint64attrvalue_stub \r\n");
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill_fusionOpParams.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);
    std::vector<std::vector<int64_t>> value;
    std::vector<int64_t> valueVec1;
    valueVec1.emplace_back(16);
    valueVec1.emplace_back(16);
    value.emplace_back(valueVec1);
    std::vector<int64_t> valueVec2;
    valueVec2.emplace_back(1);
    valueVec2.emplace_back(16);
    value.emplace_back(valueVec2);
    std::vector<int64_t> valueVec3;
    valueVec3.emplace_back(500);
    valueVec3.emplace_back(768);
    value.emplace_back(valueVec3);
    generalizedResult.dynamicJson["graphOpParams"][0]["attrs"][9]["value"] = value;
    return true;
}

void AddTensorToOpDescV3(bool isInput, std::string name, vector<int64_t> shape, Format format,  DataType data_type,
                         std::vector<std::pair<int64_t, int64_t>> &range, OpDescPtr &opDescPtr)
{
    ge::GeTensorDesc tensorDesc;
    tensorDesc.SetShape(ge::GeShape(shape));
    tensorDesc.SetOriginShape(ge::GeShape(shape));
    tensorDesc.SetShapeRange(range);
    tensorDesc.SetOriginShapeRange(range);
    tensorDesc.SetDataType(data_type);
    tensorDesc.SetOriginDataType(data_type);
    tensorDesc.SetFormat(format);
    tensorDesc.SetOriginFormat(format);
    if (isInput) {
        opDescPtr->AddInputDesc(name, tensorDesc);
    } else {
        opDescPtr->AddOutputDesc(name, tensorDesc);
    }
}
void AddOpParamToTbeOpInfoV3(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                             std::vector<std::pair<int64_t, int64_t>> &range, bool isInput, TbeOpInfo &op_info)
{
    TbeOpParam opParam;
    std::vector<TbeOpTensor> tensors;
    TbeOpTensor tensor(name, shape, dtype, format);
    tensor.SetShapeRange(range);
    tensor.SetOriginShape(shape);
    tensor.SetOriginShapeRange(range);
    tensor.SetOriginFormat(format);

    tensors.push_back(tensor);
    opParam.SetTensors(tensors);
    if (name == "input0") {
        opParam.SetValueDepend(VALUE_DEPEND_OPTIONAL);
        opParam.SetType(TT_OPT);
    }

    if (name == "input1") {
        opParam.SetValueDepend(VALUE_DEPEND_REQUIRED);
        opParam.SetType(TT_REQ);
    }

    if (isInput) {
        op_info.AddInput(opParam);
    } else {
        op_info.AddOutput(opParam);
    }
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_fusionopattrs_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    opInfo.SetBuildType(FUZZILY_BUILD);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // same as \stub\binary_stub\fusion_ops.json second bin
    (void)ge::AttrUtils::SetStr(teGraphNode[0]->GetOpDesc(), "graph_pattern", "Conv2d_Transdata_2");

    // GraphOpParams nums not match fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps
        )
        .stubs()
        .will(invoke(GeneralizeOps_graphopparamsnums_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // list_uint32 value fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps
        )
        .stubs()
        .will(invoke(GeneralizeOps_listuint32attrvalue_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps
        )
        .stubs()
        .will(invoke(GeneralizeOps_listuint32attrvalue1_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // list_string value fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps
        )
        .stubs()
        .will(invoke(GeneralizeOps_liststringattrvalue_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();

    // list_list_int64 value fail
    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps
        )
        .stubs()
        .will(invoke(GeneralizeOps_listlistint64attrvalue_stub));
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_succ) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph2(owner_graph, teGraphNode);
    std::vector<ge::Node *> teGraphNode1;
    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode1, false);
    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");
    std::shared_ptr<ge::OpDesc> opDescPtr1 = std::make_shared<ge::OpDesc>(teGraphNode1[0]->GetName(), "");
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    auto task1 = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode1, opDescPtr1, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;
    task1->buildType = FUZZILY_BUILD;
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";
    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    opInfo.SetBuildType(FUZZILY_BUILD);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    std::cout << "nodePtr:" << nodePtr->GetName().c_str() << endl;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("fill", pTbeOp);

    string binaryFileRealPath;
    res = te::fusion::BinaryManager::Instance().GetBinaryFileName(task, binaryFileRealPath, false, false);
    cout << "path:" << binaryFileRealPath << endl;
    EXPECT_EQ(res, true);
    GlobalMockObject::verify();


    TbeOpInfo opInfo2("fill", "", "Fill", "AIcoreEngine");
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo2);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo2);
    BinaryAddAttrToTbeOpInfo(opInfo2);
    std::shared_ptr<TbeOpInfo> pTbeOp2(new (std::nothrow) TbeOpInfo(opInfo2));
    opInfo2.SetBuildType(FUZZILY_BUILD);
    pTbeOp2->SetNode(nodePtr);
    std::cout << "nodePtr:" << nodePtr->GetName().c_str() << endl;
    opInfo.op_type_ = "Fill_12";
    opInfo.SetOpFileName("fill");
    std::cout << "op_name:" << opName << endl;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("fill", pTbeOp2);
    res = te::fusion::BinaryManager::Instance().GetBinaryFileName(task1, binaryFileRealPath, false, false);
    cout << "path:" << binaryFileRealPath << endl;
    EXPECT_EQ(res, true);
    GlobalMockObject::verify();

    TbeOpInfo opInfo3("matmul", "", "MatMul", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo3);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo3);
    BinaryAddAttrToTbeOpInfo(opInfo3);
    opInfo3.SetBuildType(FUZZILY_BUILD);
    std::shared_ptr<TbeOpInfo> pTbeOp3(new (std::nothrow) TbeOpInfo(opInfo3));
    pTbeOp3->SetNode(nodePtr);
    std::cout << "nodePtr:" << nodePtr->GetName().c_str() << endl;
    opInfo3.op_type_ = "Fill_12";
    opInfo3.SetOpFileName("fill12");
    std::cout << "op_name:" << opName << endl;

    TbeOpInfoCache::Instance().UpdateTbeOpInfo("matmul", pTbeOp3);
    res = te::fusion::BinaryManager::Instance().GetBinaryFileName(task1, binaryFileRealPath, false, false);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_fileinvalid_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string currentFilePath1 = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/fill.json";
    std::string path = te::fusion::RealPath(currentFilePath1);
    GeneralizedResult generalizedResult;
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, generalizedResult.dynamicJson);

    std::string jsonFilePath;
    json &binJson = generalizedResult.dynamicJson["binList"][0];
    res = te::fusion::BinaryManager::Instance().IsBinaryFileValid(false, task, binJson, jsonFilePath);
    EXPECT_EQ(res, true);
    std::string jsonFilePathCom = te::fusion::RealPath(currentFilePath) + "/" +
        "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    EXPECT_EQ(jsonFilePath, jsonFilePathCom);


    std::string stub_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    stub_file_path = te::fusion::RealPath(stub_file_path);

    jsonFilePath = stub_file_path + "/" + "stub/binary_stub/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    jsonFilePath = te::fusion::RealPath(jsonFilePath);
    printf("fuzzycompile_binary_fileinvalid_fail jsonFilePath=%s.\r\n", jsonFilePath.c_str());
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_reuse_singleop_om_success) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, true);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOmPath_.clear();
    BinaryManager::Instance().binaryOmPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOmModelPath_.clear();
    BinaryManager::Instance().binaryOmModelPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binOmAdkVersionMap_.clear();
    BinaryManager::Instance().binOmAdkVersionMap_.emplace(2, "1.73.t1.0.b010");
    BinaryManager::Instance().binOmVersionMap_.clear();
    BinaryManager::Instance().binOmVersionMap_.emplace(2, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();

    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps
        )
        .stubs()
        .will(invoke(GeneralizeOps_stub));


    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    EXPECT_EQ(res, OP_BUILD_SUCC);

    res = WaitAllFinished(graphId, tasks); // check task into pInstance->finishedTask_ quene
    EXPECT_EQ(res, false);
    // check binary json file path set ok
    std::string jsonPath;
    (void)AttrUtils::GetStr(task->outNode, "json_file_path", jsonPath);
    std::string jsonFilePath = te::fusion::RealPath(currentFilePath) + "/" +
        "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    EXPECT_EQ(jsonPath.empty(), true);
    GlobalMockObject::verify();
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_reuse_fusionop_om_success) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOmPath_.clear();
    BinaryManager::Instance().binaryOmPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOmModelPath_.clear();
    BinaryManager::Instance().binaryOmModelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binOmAdkVersionMap_.clear();
    BinaryManager::Instance().binOmAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOmVersionMap_.clear();
    BinaryManager::Instance().binOmVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    opInfo.SetBuildType(FUZZILY_BUILD);
    BinaryAddAttrToTbeOpInfo(opInfo);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();

    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps
        )
        .stubs()
        .will(invoke(GeneralizeFusionOps_stub));

    // same as \stub\binary_stub\fusion_ops.json second bin
    (void)ge::AttrUtils::SetStr(teGraphNode[0]->GetOpDesc(), "om_key_id", "te_matmul_2");

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    EXPECT_EQ(res, OP_BUILD_SUCC);

    res = WaitAllFinished(graphId, tasks); // check task into pInstance->finishedTask_ quene
    EXPECT_EQ(res, false);
    // check binary json file path set ok
    std::string jsonPath;
    (void)AttrUtils::GetStr(task->outNode, "json_file_path", jsonPath);
    std::string jsonFilePath = te::fusion::RealPath(currentFilePath) + "/" +
        "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    EXPECT_EQ(jsonPath.empty(), true);
    GlobalMockObject::verify();
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_om_key_id_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOmPath_.clear();
    BinaryManager::Instance().binaryOmPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOmModelPath_.clear();
    BinaryManager::Instance().binaryOmModelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binOmAdkVersionMap_.clear();
    BinaryManager::Instance().binOmAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOmVersionMap_.clear();
    BinaryManager::Instance().binOmVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    opInfo.SetBuildType(FUZZILY_BUILD);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // node has no "om_key_id" failed
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);

    // node "om_key_id" not matched in bin json failed
    (void)ge::AttrUtils::SetStr(teGraphNode[0]->GetOpDesc(), "om_key_id", "Conv2d_Transdata_5");
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, fuzzycompile_binary_om_singleop_fail) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOmPath_.clear();
    BinaryManager::Instance().binaryOmPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOmModelPath_.clear();
    BinaryManager::Instance().binaryOmModelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binOmAdkVersionMap_.clear();
    BinaryManager::Instance().binOmAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOmVersionMap_.clear();
    BinaryManager::Instance().binOmVersionMap_.emplace(6, "1.73.t1.0.b010");
    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    opInfo.SetOpFileName("fill_nomatch_staticKey");
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // Generalize falied
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);

    MOCKER_CPP(te::fusion::ShapeGeneralization::GeneralizeOps
        )
        .stubs()
        .will(invoke(GeneralizeFusionOps_stub));
    // statickey not match
    res = te::fusion::BinaryManager::Instance().CanReuseBinaryKernel(task);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();
}

void DeleteOfile(std::string &fileName) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    std::string filePath = te::fusion::RealPath(currentFilePath) + fileName;
    (void)remove(filePath.c_str());
}

TEST(TeFusionBinaryUTest, deleteOfile) {
    printf("deleteOfile. \r\n");
    // del te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.o
    std::string fileName = "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.o";
    DeleteOfile(fileName);

    // del te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.om
    fileName = "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.om";
    DeleteOfile(fileName);
}

TEST(TeFusionBinaryUTest, BinaryGeneralizeFusionOps_001) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV3(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV3(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV3(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("Relu","Relu");
    AddTensorToOpDescV3(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV3(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    ge::NodePtr nodeRelu = ownerGraph->AddNode(opDescPtr2);

    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("Mul1","Mul1");
    AddTensorToOpDescV3(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV3(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV3(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr3);

    ge::OpDescPtr opDescPtr4 = std::make_shared<ge::OpDesc>("Minimum","Minimum");
    AddTensorToOpDescV3(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV3(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV3(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    ge::NodePtr nodeMinimum = ownerGraph->AddNode(opDescPtr4);

    ge::OpDescPtr opDescPtr5 = std::make_shared<ge::OpDesc>("AscendQuant","AscendQuant");
    AddTensorToOpDescV3(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    AddTensorToOpDescV3(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    ge::NodePtr nodeAscendQuant = ownerGraph->AddNode(opDescPtr5);

    std::string keyStr1 = "_pattern";
    ge::AttrUtils::SetStr(nodeConv2D->GetOpDesc(), keyStr1, "x");
    std::string keyStr2 = "_pattern";
    ge::AttrUtils::SetStr(nodeRelu->GetOpDesc(), keyStr2, "x");
    std::string keyStr3 = "_pattern";
    ge::AttrUtils::SetStr(nodeMul->GetOpDesc(), keyStr3, "x");
    std::string keyStr4 = "_pattern";
    ge::AttrUtils::SetStr(nodeMinimum->GetOpDesc(), keyStr4, "x");
    std::string keyStr5 = "_pattern";
    ge::AttrUtils::SetStr(nodeAscendQuant->GetOpDesc(), keyStr5, "x");

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV3(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV3(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV3(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV3(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV3(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV3(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);

    ComputeGraphPtr ownerGraphOut = std::make_shared<ComputeGraph>("te1");
    ge::NodePtr nodeEnd = ownerGraphOut->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeRelu->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeRelu->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeMul->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeMinimum->GetOutDataAnchor(0), nodeAscendQuant->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeAscendQuant->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    TbeOpInfo op_info1("Conv2D", "", "Conv2D", "AIcoreEngine");
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info1);
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info1);
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info1);
    TbeOpInfoPtr tbeOpInfoConv2D = std::make_shared<TbeOpInfo>(op_info1);

    TbeOpInfo op_info2("Relu", "", "Relu", "AIcoreEngine");
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info2);
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info2);
    TbeOpInfoPtr tbeOpInfoRelu = std::make_shared<TbeOpInfo>(op_info2);

    TbeOpInfo op_info3("Mul1", "", "Mul1", "AIcoreEngine");
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info3);
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info3);
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info3);
    TbeOpInfoPtr tbeOpInfoMul = std::make_shared<TbeOpInfo>(op_info3);

    TbeOpInfo op_info4("Minimum", "", "Minimum", "AIcoreEngine");
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info4);
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info4);
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info4);
    TbeOpInfoPtr tbeOpInfoMinimum = std::make_shared<TbeOpInfo>(op_info4);

    TbeOpInfo op_info5("AscendQuant", "", "AscendQuant", "AIcoreEngine");
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info5);
    AddOpParamToTbeOpInfoV3({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info5);
    TbeOpInfoPtr tbeOpInfoAscendQuant = std::make_shared<TbeOpInfo>(op_info5);

    TbeAttrValue attrVal("test_attr", "strVal");
    attrVal.SetAttrSupAllFlag(true);
    //attrVal.SetAttrHasDefaultValFlag(true);

    TbeAttrValue attrVal1("test_attr1", true);
    attrVal1.SetAttrSupAllFlag(true);
    //attrVal.SetAttrHasDefaultValFlag(true);
    std::vector<int64_t> val = {1,2,3,4};
    TbeAttrValue attrVal2("test_attr2", val);
    attrVal2.SetAttrSupAllFlag(true);

    tbeOpInfoConv2D->SetPattern("formatAgnostic");
    tbeOpInfoConv2D->AddAttrValue(attrVal);
    tbeOpInfoConv2D->AddAttrValue(attrVal1);
    tbeOpInfoConv2D->AddAttrValue(attrVal2);


    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2D.get());
    teGraphNode.push_back(nodeRelu.get());
    teGraphNode.push_back(nodeMul.get());
    teGraphNode.push_back(nodeMinimum.get());
    teGraphNode.push_back(nodeAscendQuant.get());
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    auto opTaskPtr = std::make_shared<te::fusion::OpBuildTask>(
            te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                    teGraphNode, opDescPtr1, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    nlohmann::json staticJson, dynamicJson;
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(op_info1));
    opTaskPtr->pTbeOpInfo = pTbeOp;
    opTaskPtr->buildType = te::ACCURATELY_BUILD;

    std::map<std::string, std::string> options;
    options.emplace(std::make_pair("ge.opSelectImplmode", "high_performance"));
    tbeOpInfoConv2D->SetOptions(options);
    tbeOpInfoRelu->SetOptions(options);
    tbeOpInfoMul->SetOptions(options);
    tbeOpInfoMinimum->SetOptions(options);
    tbeOpInfoAscendQuant->SetOptions(options);

    tbeOpInfoConv2D->SetNode(nodeData0);
    tbeOpInfoRelu->SetNode(nodeData0);
    tbeOpInfoMul->SetNode(nodeData0);
    tbeOpInfoMinimum->SetNode(nodeData0);
    tbeOpInfoAscendQuant->SetNode(nodeData0);

    TbeOpInfoCache::Instance().UpdateTbeOpInfo("Conv2D", tbeOpInfoConv2D);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("Relu", tbeOpInfoRelu);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("Mul1", tbeOpInfoMul);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("Minimum", tbeOpInfoMinimum);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("AscendQuant", tbeOpInfoAscendQuant);
    GeneralizedResult generalizedResult;
    bool res = te::fusion::ShapeGeneralization::GeneralizeOps(opTaskPtr, generalizedResult);
    EXPECT_EQ(res, true);
}

TEST(TeFusionBinaryUTest, GetAttrGeneralizedResFromTensor_001){
    nlohmann::json tensorJson;
    tensorJson["value"] = 1;
    nlohmann::json binaryInfo;
    binaryInfo["value"] = 1;
    te::fusion::ShapeGeneralization::GetAttrGeneralizedResFromTensor(tensorJson, binaryInfo);
}

TEST(TeFusionBinaryUTest, test_fuzzy_om_reuse_path) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    setenv("ASCEND_OPP_PATH", currentFilePath.c_str(), 1);
    TeConfigInfo::Instance().InitEnvItem();
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "ascend910";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::ShortSocVersion)] = "ascend910";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::CoreType)] = "ascend910";
    const std::map<std::string, std::string> options = {
        {"op.binary.builtin", "6|/op_impl/built-in/ai_core/tbe/kernel/"},
        {"op.binary.custom", "2|/op_impl/custom/ai_core/tbe/kernel/"},
        {"om.binary.builtin", "6|/op_impl/built-in/ai_core/tbe/model/"},
        {"om.binary.custom", "2|/op_impl/custom/ai_core/tbe/model/"}
    };
    // get binary file path
    te::fusion::BinaryManager::Instance().GetBinaryOppPath(options);
    te::fusion::BinaryManager::Instance().GetAllBinaryVersionInfo(true);
    te::fusion::BinaryManager::Instance().GetAllBinaryVersionInfo(false);
}

TEST(TeFusionBinaryUTest, test_resolve_path) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    setenv("ASCEND_OPP_PATH", currentFilePath.c_str(), 1);
    TeConfigInfo::Instance().InitEnvItem();
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::ShortSocVersion)] = "ascend910";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::CoreType)] = "ascend910";
    const std::map<std::string, std::string> options = {
        {"op.binary.builtin", "6|/op_impl/built-in/ai_core/tbe/kernel/"},
        {"op.binary.custom", "100|/op_impl/custom/ai_core/tbe/kernel/, 101|/op_impl/custom/ai_core/tbe/kernel/"},
        {"om.binary.builtin", "/op_impl/built-in/ai_core/tbe/model/"},
        {"om.binary.custom_fail", "2|/op_impl/custom/ai_core/tbe/model/"},
        {"om.binary.custom", "100|/op_impl/custom/ai_core/tbe/model/"}
    };
    // get binary file path
    te::fusion::BinaryManager::Instance().binaryOppPath_.clear();
    te::fusion::BinaryManager::Instance().binaryOppKernelPath_.clear();
    te::fusion::BinaryManager::Instance().binaryOmPath_.clear();
    te::fusion::BinaryManager::Instance().binaryOmModelPath_.clear();
    te::fusion::BinaryManager::Instance().GetBinaryOppPath(options);
    te::fusion::BinaryManager::Instance().GetAllBinaryVersionInfo(true);
    te::fusion::BinaryManager::Instance().GetAllBinaryVersionInfo(false);
    size_t ret = te::fusion::BinaryManager::Instance().binaryOppPath_.size();
}

TEST(TeFusionBinaryUTest, test_fuzzy_om_reuse_version) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    setenv("ASCEND_OPP_PATH", currentFilePath.c_str(), 1);
    TeConfigInfo::Instance().InitEnvItem();
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "ascend910";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::CoreType)] = "ascend910";

    // get binary file version and cur version
    std::string filePath = te::fusion::RealPath(currentFilePath) + "/";
    std::string binAdkVersion;
    std::string binOppVersion;
    te::fusion::BinaryManager::Instance().GetBinaryVersionInfo(filePath,
        binAdkVersion, binOppVersion);
    std::string binCustomAdkVersion;
    std::string binCustomOppVersion;
    te::fusion::BinaryManager::Instance().GetBinaryVersionInfo(filePath,
        binCustomAdkVersion, binCustomOppVersion);

    // check version
    std::string adkVersion = "1.73.t1.0.b010"; // same as file version.info
    std::string oppVersion = "1.73.t1.0.b010";
    EXPECT_EQ(binAdkVersion, adkVersion);
    EXPECT_EQ(binOppVersion, oppVersion);
    EXPECT_EQ(binCustomAdkVersion, adkVersion);
    EXPECT_EQ(binCustomOppVersion, oppVersion);
}

TEST(TeFusionBinaryUTest, test_fuzzy_om_compile_singleop_success) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, true);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOmPath_.clear();
    BinaryManager::Instance().binaryOmPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOmModelPath_.clear();
    BinaryManager::Instance().binaryOmModelPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binOmAdkVersionMap_.clear();
    BinaryManager::Instance().binOmAdkVersionMap_.emplace(2, "1.73.t1.0.b010");
    BinaryManager::Instance().binOmVersionMap_.clear();
    BinaryManager::Instance().binOmVersionMap_.emplace(2, "1.73.t1.0.b010");

    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(2, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(2, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(2, "1.73.t1.0.b010");

    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";
    ge::GetThreadLocalContext().global_options_[ge::BUILD_INNER_MODEL] = "false";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    BinaryAddAttrToTbeOpInfo(opInfo);
    TbeAttrValue attr_var("_input_name_key", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    opInfo.AddAttrValue(attr_var);

    OpDescPtr opdesc1;
    std::string opType = "Fill";
    vector<string> inputnameList;
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    FillTensorDesc(input_desc1, {16, 16}, DT_FLOAT16);
    FillTensorDesc(input_desc2, {16, 16}, DT_FLOAT16);
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, {16, 16}, DT_FLOAT16);
    FillTensorDesc(output_desc2, {16, 16}, DT_FLOAT16);
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);

    opdesc1 = CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opdesc1, "variable_attr", variableValue);

    NodePtr nodePoint1 = owner_graph->AddNode(opdesc1);
    opInfo.SetNode(nodePoint1);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opdesc1, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    printf("res is %d.", res);
    EXPECT_EQ(res, OP_BUILD_SUCC);
}

TEST(TeFusionBinaryUTest, test_fuzzy_om_compile_fusionop_success) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode, false);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";

    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");

    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";
    ge::GetThreadLocalContext().global_options_[ge::BUILD_INNER_MODEL] = "false";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    opInfo.SetBuildType(FUZZILY_BUILD);
    BinaryAddAttrToTbeOpInfo(opInfo);
    TbeAttrValue attr_var("_input_name_key", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    opInfo.AddAttrValue(attr_var);

    OpDescPtr opdesc1;
    std::string opType = "Fill";
    vector<string> inputnameList;
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    FillTensorDesc(input_desc1, {16, 16}, DT_FLOAT16);
    FillTensorDesc(input_desc2, {16, 16}, DT_FLOAT16);
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, {16, 16}, DT_FLOAT16);
    FillTensorDesc(output_desc2, {16, 16}, DT_FLOAT16);
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);

    opdesc1 = CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opdesc1, "variable_attr", variableValue);

    NodePtr nodePoint1 = owner_graph->AddNode(opdesc1);
    opInfo.SetNode(nodePoint1);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opdesc1, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);
    task->pTbeOpInfo = pTbeOp;

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    printf("res is %d.", res);
    EXPECT_EQ(res, OP_BUILD_SUCC);
}

TEST(TeFusionBinaryUTest, test_check_git_compile_success) {
    std::string opType = "Fill";
    vector<string> inputnameList0;
    inputnameList0.push_back("data_0");
    inputnameList0.push_back("data_0:1");

    vector<GeTensorDesc> inputdescList0;
    vector<GeTensorDesc> outputdescList0;
    GeTensorDesc input_desc1_0;
    GeTensorDesc input_desc2_0;
    FillTensorDesc(input_desc1_0, {16, 16}, DT_FLOAT16);
    FillTensorDesc(input_desc2_0, {16, 16}, DT_FLOAT16);
    inputdescList0.push_back(input_desc1_0);
    inputdescList0.push_back(input_desc2_0);

    OpDescPtr opdesc_0;
    string opNameTest = "test";
    opdesc_0 = CreateOpDesc(opNameTest, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList0, 2, inputdescList0, outputdescList0);

    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraph(owner_graph, teGraphNode, false);

    ComputeGraphPtr owner_graph1 = std::make_shared<ComputeGraph>("te");
    BinaryTeFusionCreateSingleNodeGraph(owner_graph1, teGraphNode, false);

    string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");

    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";
    ge::GetThreadLocalContext().global_options_[ge::BUILD_INNER_MODEL] = "false";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    opInfo.SetBuildType(FUZZILY_BUILD);
    BinaryAddAttrToTbeOpInfo(opInfo);
    TbeAttrValue attr_var("_input_name_key", (int8_t)1);
    opInfo.AddAttrValue(attr_var);

    OpDescPtr opdesc1;
    opType = "Fill";
    vector<string> inputnameList;
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    FillTensorDesc(input_desc1, {16, 16}, DT_FLOAT16);
    FillTensorDesc(input_desc2, {16, 16}, DT_FLOAT16);
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, {16, 16}, DT_FLOAT16);
    FillTensorDesc(output_desc2, {16, 16}, DT_FLOAT16);
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);

    opdesc1 = CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opdesc1, "variable_attr", variableValue);
    ComputeGraphPtr graph_0 = std::make_shared<ComputeGraph>("te");

    NodePtr nodePoint1 = graph_0->AddNode(opdesc_0);
    NodePtr nodePoint2 = graph_0->AddNode(opdesc1);
    opInfo.SetNode(nodePoint1);
    opInfo.SetIsUnknownShape(true);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opdesc1, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("session_id_" + opNameTest, pTbeOp);
    task->pTbeOpInfo = pTbeOp;
    //ge::Node node_0 = new ge::Node(opdesc_0, owner_graph);
    ge::Node *node_tmp = nodePoint1.get();
    ge::Node *node_tmp2 = nodePoint2.get();
    task->opNodes.clear();
    task->opNodes.push_back(node_tmp);
    bool bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, false);
    ge::GetThreadLocalContext().global_options_[ge::JIT_COMPILE] = "0";
    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, true);
    task->opNodes.clear();
    task->opNodes.push_back(node_tmp2);
    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, false);

    task->opNodes.clear();
    task->opNodes.push_back(node_tmp);
    task->opNodes.push_back(node_tmp2);
    ge::AttrUtils::SetBool(opdesc_0, "_unknown_shape", true);
    ge::AttrUtils::SetBool(opdesc1, "_unknown_shape", true);
    std::string keyName1;
    TbeOpInfoCache::Instance().GetOpKeyNameByNode(node_tmp, keyName1);
    std::string keyName2;
    TbeOpInfoCache::Instance().GetOpKeyNameByNode(node_tmp2, keyName2);
    std::shared_ptr<TbeOpInfo> op_info_ptr1 =
        std::make_shared<te::TbeOpInfo>(nodePoint1->GetName(), "", nodePoint1->GetType(), "");
    std::shared_ptr<TbeOpInfo> op_info_ptr2 =
        std::make_shared<te::TbeOpInfo>(nodePoint2->GetName(), "", nodePoint2->GetType(), "");
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(keyName1, op_info_ptr1);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(keyName2, op_info_ptr2);
    bool needCheckNext;
    bool ret = te::fusion::BinaryManager::Instance().GetOpJitCompileForBinReuse(task, needCheckNext);
    EXPECT_EQ(ret, false);
    op_info_ptr1->SetIsUnknownShape(true);
    op_info_ptr2->SetIsUnknownShape(true);
    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, true);

    op_info_ptr1->SetOpJitCompile(JitCompileType::REUSE_BINARY);
    op_info_ptr2->SetOpJitCompile(JitCompileType::REUSE_BINARY);
    ret = te::fusion::BinaryManager::Instance().GetOpJitCompileForBinReuse(task, needCheckNext);
    EXPECT_EQ(ret, true);
    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, true);

    op_info_ptr1->SetOpJitCompile(JitCompileType::ONLINE);
    op_info_ptr2->SetOpJitCompile(JitCompileType::ONLINE);
    ret = te::fusion::BinaryManager::Instance().GetOpJitCompileForBinReuse(task, needCheckNext);
    EXPECT_EQ(ret, false);
    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, false);

    op_info_ptr1->SetOpJitCompile(JitCompileType::DEFAULT);
    op_info_ptr2->SetOpJitCompile(JitCompileType::DEFAULT);
    ge::GetThreadLocalContext().global_options_[ge::JIT_COMPILE] = "1";
    ret = te::fusion::BinaryManager::Instance().GetOpJitCompileForBinReuse(task, needCheckNext);
    EXPECT_EQ(ret, false);

    op_info_ptr1->SetOpJitCompile(JitCompileType::STATIC_BINARY_DYNAMIC_ONLINE);
    op_info_ptr2->SetOpJitCompile(JitCompileType::STATIC_BINARY_DYNAMIC_ONLINE);
    ge::GetThreadLocalContext().global_options_[ge::JIT_COMPILE] = "1";
    ret = te::fusion::BinaryManager::Instance().GetOpJitCompileForBinReuse(task, needCheckNext);
    EXPECT_EQ(ret, false);
    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, false);

    op_info_ptr1->SetOpJitCompile(JitCompileType::STATIC_BINARY_DYNAMIC_BINARY);
    op_info_ptr2->SetOpJitCompile(JitCompileType::STATIC_BINARY_DYNAMIC_BINARY);
    ge::GetThreadLocalContext().global_options_[ge::JIT_COMPILE] = "1";
    ret = te::fusion::BinaryManager::Instance().GetOpJitCompileForBinReuse(task, needCheckNext);
    EXPECT_EQ(ret, true);
    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, true);

    ge::GetThreadLocalContext().global_options_[ge::JIT_COMPILE] = "0";
    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, true);

    bres = te::fusion::BinaryManager::Instance().CheckJitCompileForBinReuse(task);
    EXPECT_EQ(bres, true);
}

TEST(TeFusionBinaryUTest, test_check_spec_shape_reuse_success) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    BinaryTeFusionCreateSingleNodeGraphV2(owner_graph, teGraphNode, false, 0);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = ACCURATELY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    BinaryManager::Instance().binaryOppPath_.clear();
    BinaryManager::Instance().binaryOppPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binaryOppKernelPath_.clear();
    BinaryManager::Instance().binaryOppKernelPath_.emplace(6, te::fusion::RealPath(currentFilePath) + "/");
    BinaryManager::Instance().binAdkVersionMap_.clear();
    BinaryManager::Instance().binAdkVersionMap_.emplace(6, "1.73.t1.0.b010");
    BinaryManager::Instance().binOppVersionMap_.clear();
    BinaryManager::Instance().binOppVersionMap_.emplace(6, "1.73.t1.0.b010");

    TeConfigInfo::Instance().adkVersion_ = "1.73.t1.0.b010";
    TeConfigInfo::Instance().oppVersion_ = "1.73.t1.0.b010";
    ge::GetThreadLocalContext().global_options_[ge::BUILD_INNER_MODEL] = "true";

    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // name as \stub\binary_stub\json file name
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    BinaryAddOpParamToTbeOpInfo({1,2}, "float16", opName, "input0", range, true, opInfo);
    BinaryAddOpParamToTbeOpInfo({1,2}, "int64", opName, "output0", range, false, opInfo);
    opInfo.SetBuildType(ACCURATELY_BUILD);
    BinaryAddAttrToTbeOpInfo(opInfo);
    TbeAttrValue attr_var("_dst_type", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    opInfo.AddAttrValue(attr_var);

    OpDescPtr opdesc1;
    std::string opType = "Fill";
    vector<string> inputnameList;
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    FillTensorDesc(input_desc1, {-1, -1}, DT_FLOAT16);
    FillTensorDesc(input_desc2, {-1, -1}, DT_FLOAT16);
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, {-1, -1}, DT_FLOAT16);
    FillTensorDesc(output_desc2, {-1, -1}, DT_FLOAT16);
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);

    opdesc1 = CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opdesc1, "variable_attr", variableValue);

    NodePtr nodePoint1 = owner_graph->AddNode(opdesc1);
    opInfo.SetNode(nodePoint1);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opdesc1, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);
    pTbeOp->SetBuildType(ACCURATELY_BUILD);
    task->pTbeOpInfo = pTbeOp;

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    printf("res is %d.", res);
    EXPECT_EQ(res, OP_BUILD_SUCC);
    auto geGraphOptions = ge::GetThreadLocalContext().GetAllGraphOptions();
    geGraphOptions.emplace(ge::OP_DEBUG_OPTION, "oom");
    ge::GetThreadLocalContext().SetGraphOption(geGraphOptions);
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
}

TEST(TeFusionBinaryUTest, test_scalar_cannot_reuse) {
    ge::GeTensorDescPtr tensor = std::make_shared<ge::GeTensorDesc>();
    bool bres = te::fusion::BinaryManager::Instance().IsAllDimsDyn(tensor);
    EXPECT_EQ(bres, false);
}

TEST(TeFusionBinaryUTest, test_MatchStaticKeyWithOptionalInputNull) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    bool hasOmKeyId = false;
    std::vector<ge::Node *> teGraphNode;
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
        teGraphNode, nullptr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    GeneralizedResult generalizedResult;

    std::string currentFilePath1 = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/conv2d_transpose_op.json";
    std::string path1 = te::fusion::RealPath(currentFilePath1);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path1, generalizedResult.staticJson);

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/conv2d_transpose.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    nlohmann::json binListJson;
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, binListJson);
    bool bres = te::fusion::BinaryManager::Instance().MatchStaticKeyWithOptionalInputNull(task, binListJson, generalizedResult);
    EXPECT_EQ(bres, true);

    std::string currentFilePath2 = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/conv2d_transpose_fail.json";
    std::string path2 = te::fusion::RealPath(currentFilePath2);
    nlohmann::json binListJson1;
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path2, binListJson1);
    bres = te::fusion::BinaryManager::Instance().MatchStaticKeyWithOptionalInputNull(task, binListJson1, generalizedResult);
    EXPECT_EQ(bres, false);
}

bool GetBinaryInfoConfigPath_Stub(te::fusion::BinaryInfoBase *This, const string &configPath, std::string &configRealPath,
                                  bool isSuperKernel, const string &path)
{
    printf("GetBinaryInfoConfigPath_Stub \r\n");

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/ai_core/tbe/config/binary_info_config.json";
    configRealPath = currentFilePath;
    return true;
}

TEST(TeFusionBinaryUTest, test_DtypeNormalize) {
    GenerateSimpleKey generateKey = {"PadV2", SimpleKeyModeType::COMPATIBLE_MODE, "high_performance", "false", true};
    DtypeFormatMode dtypeFormatMode;
    dtypeFormatMode.dTypeMode = {"bit"};
    int index = 0;
    std::string dtype = "float6_e3m2";
    generateKey.DtypeNormalize(index, dtypeFormatMode, dtype);
    dtype = "float6_e3m2";
    generateKey.DtypeNormalize(index, dtypeFormatMode, dtype);
}

TEST(TeFusionBinaryUTest, test_ParseBinaryInfo) {
    MOCKER_CPP(&te::fusion::BinaryInfoBase::GetBinaryInfoConfigPath,
        bool(te::fusion::BinaryInfoBase::*)(const std::string &, std::string &, const bool&, const std::string&) const)
        .stubs()
        .will(invoke(GetBinaryInfoConfigPath_Stub));


    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::ShortSocVersion)] = "910B";
    SimpleKeyModeType simpleKeyMode = SimpleKeyModeType::SIMPLE_MODE;
    DtypeFormatMode inputMode;
    DtypeFormatMode outputMode;
    int optionalInputMode;
    DynamicParamModeEnum dynamicMode;
    std::string simpleKey = "Add/d=0,p=0/0,2/0,2/0,2";
    std::string jsonFilePath;
    BinaryInfoBasePtr binaryInfoPtr = std::make_shared<te::fusion::BinaryInfoBase>();
    binaryInfoPtr = std::make_shared<BinaryInfoBase>();
    std::string configJsonPath = "";
    bool res = binaryInfoPtr->ParseBinaryInfoFile(configJsonPath, false);
    EXPECT_EQ(res, true);
    res = binaryInfoPtr->GetSimpleKeyMode("Add", simpleKeyMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(simpleKeyMode, SimpleKeyModeType::SIMPLE_MODE);
    res = binaryInfoPtr->GetInputMode("Add", inputMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(inputMode.count, 2);
    res = binaryInfoPtr->GetOutputMode("Add", outputMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(outputMode.count, 1);
    res = binaryInfoPtr->GetOptionalInputMode("Add", optionalInputMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(optionalInputMode, 0);
    res = binaryInfoPtr->GetDynParamModeByOpType("Add", dynamicMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(dynamicMode, DynamicParamModeEnum::FOLDED_WITH_DESC);

    res = binaryInfoPtr->MatchSimpleKey("Add", simpleKey, jsonFilePath);
    EXPECT_EQ(res, true);
    EXPECT_EQ(jsonFilePath, "ascend910/add/Add_ee98c6628030785f610b924ab1557b31_high_performance.json");

    res = binaryInfoPtr->GetSimpleKeyMode("aipp", simpleKeyMode);
    EXPECT_EQ(res, false);
    EXPECT_EQ(simpleKeyMode, SimpleKeyModeType::SIMPLE_MODE);
    res = binaryInfoPtr->GetInputMode("aipp", inputMode);
    EXPECT_EQ(res, false);
    EXPECT_EQ(inputMode.count, 2);
    res = binaryInfoPtr->GetOutputMode("aipp", outputMode);
    EXPECT_EQ(res, false);
    EXPECT_EQ(outputMode.count, 1);
    res = binaryInfoPtr->GetOptionalInputMode("aipp", optionalInputMode);
    EXPECT_EQ(res, false);
    EXPECT_EQ(optionalInputMode, 0);
    res = binaryInfoPtr->GetDynParamModeByOpType("Aipp", dynamicMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(dynamicMode, DynamicParamModeEnum::UNFOLDED);
    res = binaryInfoPtr->MatchSimpleKey("aipp", simpleKey, jsonFilePath);
    EXPECT_EQ(res, false);

    res = binaryInfoPtr->GetSimpleKeyMode("AccumulateNV2", simpleKeyMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(simpleKeyMode, SimpleKeyModeType::COMPATIBLE_MODE);
    res = binaryInfoPtr->GetInputMode("AccumulateNV2", inputMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(inputMode.count, 1);
    res = binaryInfoPtr->GetOutputMode("AccumulateNV2", outputMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(outputMode.count, 1);
    res = binaryInfoPtr->GetOptionalInputMode("AccumulateNV2", optionalInputMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(optionalInputMode, 0);
    res = binaryInfoPtr->MatchSimpleKey("AccumulateNV2", simpleKey, jsonFilePath);
    EXPECT_EQ(res, false);

    res = binaryInfoPtr->GetDynParamModeByOpType("null", dynamicMode);
    EXPECT_EQ(res, false);

    res = binaryInfoPtr->GetSimpleKeyMode("ConcatD", simpleKeyMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(simpleKeyMode, SimpleKeyModeType::COMPATIBLE_MODE);
    res = binaryInfoPtr->GetSimpleKeyMode("Abs", simpleKeyMode);
    EXPECT_EQ(res, true);
    EXPECT_EQ(simpleKeyMode, SimpleKeyModeType::SIMPLE_MODE);
    res = binaryInfoPtr->GetSimpleKeyMode("test", simpleKeyMode);
    EXPECT_EQ(res, false);
    res = binaryInfoPtr->GetSimpleKeyMode("StridedSliceV3", simpleKeyMode);
    EXPECT_EQ(res, false);

    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    TbeOpInfo opInfo("padV2", "", "PadV2", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int64", "ND", "input0", range, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int32", "ND", "input1", range, true, pTbeOp, TT_DYN);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int16", "ND", "input2", range, true, pTbeOp, TT_OPT);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int8", "ND", "output", range, false, pTbeOp, TT_OPT);
    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("padV2", pTbeOp);
    std::string strVal = "reflect";
    TbeAttrValue attr_var("mode", strVal);
    pTbeOp->AddAttrValue(attr_var);
    std::vector<TbeOpParam> inputs;
    std::vector<TbeOpParam> outputs;
    std::vector<TbeAttrValue> attrValues;
    pTbeOp->GetInputs(inputs);
    pTbeOp->GetOutputs(outputs);
    pTbeOp->GetAttrValues(attrValues);
    GenerateSimpleKey generateKey = {"PadV2", SimpleKeyModeType::COMPATIBLE_MODE, "high_performance", "false", true};
    generateKey.SetInputs(inputs);
    generateKey.SetOutputs(outputs);
    generateKey.SetAttrs(attrValues);
    generateKey.SetBinaryInfoPtr(binaryInfoPtr);
    std::string simpleKeystr;
    res = generateKey.GenerateSimpleKeyStr(simpleKeystr);
    EXPECT_EQ(res, true);
    EXPECT_EQ(simpleKeystr, "PadV2/d=0,p=1/9,2/3,2/6,2/2,2/reflect");

    TbeOpInfoPtr pTbeOp1 = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int8", "ND", "input0", range, true, pTbeOp1, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int32", "ND", "input1", range, true, pTbeOp1, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "bool", "ND", "input2", range, true, pTbeOp1, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int8", "ND", "output", range, false, pTbeOp1, TT_REQ);
    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("padV2", pTbeOp1);
    pTbeOp1->AddAttrValue(attr_var);
    pTbeOp1->GetInputs(inputs);
    pTbeOp1->GetOutputs(outputs);
    pTbeOp1->GetAttrValues(attrValues);
    GenerateSimpleKey generateKey1 = {"PadV2", SimpleKeyModeType::COMPATIBLE_MODE, "high_performance", "false", false};
    generateKey1.SetInputs(inputs);
    generateKey1.SetOutputs(outputs);
    generateKey1.SetAttrs(attrValues);
    generateKey1.SetBinaryInfoPtr(binaryInfoPtr);
    std::string simpleKeystr1;
    res = generateKey1.GenerateSimpleKeyStr(simpleKeystr1);
    EXPECT_EQ(res, true);
    EXPECT_EQ(simpleKeystr1, "PadV2/d=0,p=1/2,2/3,2/2,2/2,2/reflect");
}

TEST(TeFusionBinaryUTest, test_GetSimplifiedKey) {
    ge::AscendString example("test");
    optiling::GEN_SIMPLIFIEDKEY_FUNC pFunc;
    pFunc = testFunc;
    optiling::OpCheckFuncRegistry::RegisterGenSimplifiedKeyFunc(example, pFunc);
    
    ge::AscendString errName("notExisted");
    auto nullFunc = optiling::OpCheckFuncRegistry::GetGenSimplifiedKeyFun(errName);
    EXPECT_EQ(nullFunc, nullptr);

    auto func = optiling::OpCheckFuncRegistry::GetGenSimplifiedKeyFun(example);
    EXPECT_NE(func, nullptr);
}

TEST(TeFusionBinaryUTest, test_GetInfoFailed) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;

    std::vector<ge::Node *> teGraphNode;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("mul", "Mul");
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{1,10}};
    AddTensorToOpDesc(true, "input0", {-1,-1}, FORMAT_NCHW, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, FORMAT_NCHW, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(false, "output", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opDescPtr, "variable_attr", variableValue);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, nullptr);
    teGraphNode.push_back(nodePtr.get());
    std::string opName = teGraphNode[0]->GetName();

    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    TeFusionManager *pInstance = te::fusion::TeFusionManager::GetInstance();
    std::string jsonFilePath;
    res = te::fusion::BinaryManager::Instance().MatchSimplifiedKey(opTask, jsonFilePath);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, test_GenerateSimpleKeyStrInputOutputFailed) {
    TeCacheSpaceManager &fusion_serial = TeCacheSpaceManager::Instance();

    MOCKER_CPP(&te::fusion::BinaryInfoBase::GetBinaryInfoConfigPath,
        bool(te::fusion::BinaryInfoBase::*)(const std::string &, std::string &, const bool&, const std::string&) const)
        .stubs()
        .will(invoke(GetBinaryInfoConfigPath_Stub));


    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::ShortSocVersion)] = "910B";
    BinaryInfoBasePtr binaryInfoPtr = std::make_shared<te::fusion::BinaryInfoBase>();
    binaryInfoPtr = std::make_shared<BinaryInfoBase>();
    std::string configJsonPath = "";
    bool res = binaryInfoPtr->ParseBinaryInfoFile(configJsonPath, false);
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    TbeOpInfo opInfo("padV2", "", "PadV2", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int64", "ND", "input0", range, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int32", "ND", "input1", range, true, pTbeOp, TT_DYN);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int16", "ND", "input2", range, true, pTbeOp, TT_OPT);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int8", "ND", "output", range, false, pTbeOp, TT_OPT);
    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("padV2", pTbeOp);
    std::string strVal = "reflect";
    TbeAttrValue attr_var("mode", strVal);
    pTbeOp->AddAttrValue(attr_var);
    std::vector<TbeOpParam> inputs;
    std::vector<TbeOpParam> outputs;
    std::vector<TbeAttrValue> attrValues;
    pTbeOp->GetInputs(inputs);
    pTbeOp->GetOutputs(outputs);
    pTbeOp->GetAttrValues(attrValues);

    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("mul", "Mul");
    AddTensorToOpDesc(true, "input0", {-1,-1}, FORMAT_NCHW, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, FORMAT_NCHW, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, nullptr);
    pTbeOp->SetNode(nodePtr);

    GenerateSimpleKey generateKey = {"PadV2", SimpleKeyModeType::CUSTOM_MODE, "high_performance", "false", true};
    
    generateKey.SetAttrs(attrValues);
    generateKey.SetBinaryInfoPtr(binaryInfoPtr);
    generateKey.SetOpInfoPtr(pTbeOp);
    std::string simpleKeystr;
    res = generateKey.GenerateSimpleKeyStr(simpleKeystr);
    EXPECT_EQ(res, false);
    generateKey.SetInputs(inputs);
    res = generateKey.GenerateSimpleKeyStr(simpleKeystr);
    EXPECT_EQ(res, false);
    generateKey.SetOutputs(outputs);
    res = generateKey.GenerateSimpleKeyStr(simpleKeystr);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, test_CustomMode) {
    TeCacheSpaceManager &fusion_serial = TeCacheSpaceManager::Instance();

    MOCKER_CPP(&te::fusion::BinaryInfoBase::GetBinaryInfoConfigPath,
        bool(te::fusion::BinaryInfoBase::*)(const string &, std::string &, const bool&, const std::string&) const)
        .stubs()
        .will(invoke(GetBinaryInfoConfigPath_Stub));

    MOCKER(optiling::OpCheckFuncRegistry::GetGenSimplifiedKeyFun)
        .stubs()
        .will(returnValue(optiling::GEN_SIMPLIFIEDKEY_FUNC(&GenSimplifiedkeyStub)));


    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::ShortSocVersion)] = "910B";
    TeFusionManager *pInstance = te::fusion::TeFusionManager::GetInstance();
    MOCKER_CPP(&te::fusion::BinaryInfoBase::GetBinaryInfoConfigPath,
        bool(te::fusion::BinaryInfoBase::*)(const string &, std::string &, const bool&, const std::string&) const)
        .stubs()
        .will(invoke(GetBinaryInfoConfigPath_Stub));
    BinaryManager::Instance().binaryConfigPathMap_ = {{6, ""}};
    BinaryManager::Instance().ParseAllBinaryInfoConfigPath();
    BinaryInfoBasePtr binaryInfoPtr = BinaryManager::Instance().binaryInfoPtrMap_[6];
    std::string configJsonPath = "";
    bool res = binaryInfoPtr->ParseBinaryInfoFile(configJsonPath, false);
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    TbeOpInfo opInfo("padV2", "", "PadV2", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int64", "ND", "input0", range, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int32", "ND", "input1", range, true, pTbeOp, TT_DYN);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int16", "ND", "input2", range, true, pTbeOp, TT_OPT);
    AddOpParamToTbeOpInfoPtr({-1, -1}, "int8", "ND", "output", range, false, pTbeOp, TT_OPT);
    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("padV2", pTbeOp);
    std::string strVal = "reflect";
    TbeAttrValue attr_var("mode", strVal);
    pTbeOp->AddAttrValue(attr_var);
    std::vector<TbeOpParam> inputs;
    std::vector<TbeOpParam> outputs;
    std::vector<TbeAttrValue> attrValues;
    pTbeOp->GetInputs(inputs);
    pTbeOp->GetOutputs(outputs);
    pTbeOp->GetAttrValues(attrValues);
    GenerateSimpleKey generateKeyTest1 = {"PadV2", SimpleKeyModeType::SIMPLE_MODE, "test", "false", true};
    res = generateKeyTest1.CheckParamSetDefaultVal();
    EXPECT_EQ(res, false);
    GenerateSimpleKey generateKeyTest2 = {"PadV2", SimpleKeyModeType::SIMPLE_MODE, "high_performance", "test", true};
    res = generateKeyTest2.CheckParamSetDefaultVal();
    EXPECT_EQ(res, false);
    GenerateSimpleKey generateKeyTest3 = {"", SimpleKeyModeType::SIMPLE_MODE, "high_performance", "test", true};
    res = generateKeyTest2.CheckParamSetDefaultVal();
    EXPECT_EQ(res, false);
    GenerateSimpleKey generateKey = {"PadV2", SimpleKeyModeType::CUSTOM_MODE, "high_performance", "false", true};
    generateKey.SetInputs(inputs);
    generateKey.SetOutputs(outputs);
    generateKey.SetAttrs(attrValues);
    generateKey.SetBinaryInfoPtr(binaryInfoPtr);
    generateKey.SetOpInfoPtr(pTbeOp);

    std::string simpleKeystr;
    res = generateKey.GenerateSimpleKeyStr(simpleKeystr);
    EXPECT_EQ(res, false);

    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("pad", "PadV2");
    AddTensorToOpDesc(true, "input0", {-1,-1}, ge::Format::FORMAT_ND, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, ge::Format::FORMAT_ND, DT_INT32, range, opDescPtr);
    AddTensorToOpDesc(true, "input2", {-1,-1}, ge::Format::FORMAT_ND, DT_INT16, range, opDescPtr);
    AddTensorToOpDesc(false, "output", {-1,-1}, ge::Format::FORMAT_ND, DT_INT64, range, opDescPtr);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, nullptr);
    pTbeOp->SetNode(nodePtr);

    // gert::OpImplSpaceRegistryV2Ptr spaceRegistryStub = std::make_shared<gert::OpImplSpaceRegistryV2>();
    // auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
    // gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
    // op_impl_func.gen_simplifiedkey = GenSimplifiedKeyKernelFuncStub;
    // registry_holder->AddTypesToImpl("PadV2", op_impl_func);
    // spaceRegistryStub->AddRegistry(registry_holder);

    // gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(spaceRegistryStub);
    // std::string str = "PadV2/d=0,p=0";
    // bool bres = generateKey.GenerateCustomizeSimplifiedKey(str);
    // EXPECT_EQ(bres, true);
    // EXPECT_EQ(str, "PadV2/d=0,p=0/p=[null,null]/float");
}

TEST(TeFusionBinaryUTest, test_MatchSimpleKey) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("mul", "Mul");
    ge::AttrUtils::SetInt(opDescPtr, "_fe_imply_type", 6);
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{1,10}};
    AddTensorToOpDesc(true, "input0", {-1,-1}, FORMAT_NCHW, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, FORMAT_NCHW, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(false, "output", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opDescPtr, "variable_attr", variableValue);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    teGraphNode.push_back(nodePtr.get());
    std::string opName = teGraphNode[0]->GetName();

    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    opTask->kernel = "te_mul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";
    opTask->buildType = FUZZILY_BUILD;
    opTask->newCompile = true;
    opTask->kernel = "te_mul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    TbeOpInfo opInfo("mul", "", "Mul", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "int64", "NCHW", "input0", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "int64", "NCHW", "input1", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({9,2,3,4}, "int64", "NHWC", "output", range, false, pTbeOp, TT_REQ);

    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    TbeAttrValue attr_var("_input_name_key", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    pTbeOp->AddAttrValue(attr_var);
    TbeAttrValue attr_var_1("_dst_type", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    pTbeOp->AddAttrValue(attr_var_1);
    pTbeOp->SetNode(nodePtr);

    std::map<std::string, std::string> options;
    options["deterministic"] = "true";
    (void)pTbeOp->SetOptions(options);
    pTbeOp->SetOpImplMode("default");

    TeFusionManager *pInstance = te::fusion::TeFusionManager::GetInstance();
    MOCKER_CPP(&te::fusion::BinaryInfoBase::GetBinaryInfoConfigPath,
        bool(te::fusion::BinaryInfoBase::*)(const string &, std::string &, const bool&, const std::string&) const)
        .stubs()
        .will(invoke(GetBinaryInfoConfigPath_Stub));
    BinaryManager::Instance().binaryConfigPathMap_ = {{6, ""}};
    BinaryManager::Instance().ParseAllBinaryInfoConfigPath();
    BinaryInfoBasePtr binaryInfoPtr = BinaryManager::Instance().binaryInfoPtrMap_[6];
    std::string jsonFilePath;
    res = te::fusion::BinaryManager::Instance().MatchSimplifiedKey(opTask, jsonFilePath);
    EXPECT_EQ(res, true);
    EXPECT_EQ(jsonFilePath, "ascend910/mul/Mul_41dadce325b0f810d03359af2a38990b_high_performance.json");

    res = te::fusion::BinaryManager::Instance().ReuseBinKernelBySimpleKey(opTask);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, test_MatchSimpleKey2) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("padV3Grad", "PadV3Grad");
    ge::AttrUtils::SetInt(opDescPtr, "_fe_imply_type", 6);
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{1,10}};
    AddTensorToOpDesc(true, "input0", {-1,-1}, ge::Format::FORMAT_ND, DT_FLOAT16, range1, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, ge::Format::FORMAT_ND, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(false, "output", {9,2,3,4}, ge::Format::FORMAT_ND, DT_FLOAT16, range, opDescPtr);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opDescPtr, "variable_attr", variableValue);
    // ge::AttrUtils::SetInt(opDescPtr, "_ascendc_sp_sub_id", 1);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    teGraphNode.push_back(nodePtr.get());
    std::string opName = teGraphNode[0]->GetName();

    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    opTask->kernel = "te_padV3Grad_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    opTask->buildType = FUZZILY_BUILD;
    opTask->newCompile = true;
    opTask->kernel = "te_padV3Grad_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    TbeOpInfo opInfo("padV3Grad", "", "PadV3Grad", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "float16", "ND", "input0", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "int64", "ND", "input1", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({9,2,3,4}, "float16", "ND", "output", range, false, pTbeOp, TT_REQ);

    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strVal = "reflect";
    TbeAttrValue attr_var("mode", strVal);
    pTbeOp->AddAttrValue(attr_var);

    TbeAttrValue attr_var_1("paddings_contiguous", false);
    pTbeOp->AddAttrValue(attr_var_1);
    pTbeOp->SetNode(nodePtr);

    std::map<std::string, std::string> options;
    options["deterministic"] = "false";
    (void)pTbeOp->SetOptions(options);
    pTbeOp->SetOpImplMode("default");

    MOCKER_CPP(&te::fusion::BinaryInfoBase::GetBinaryInfoConfigPath,
        bool(te::fusion::BinaryInfoBase::*)(const string &, std::string &, const bool&, const std::string&) const)
        .stubs()
        .will(invoke(GetBinaryInfoConfigPath_Stub));

    MOCKER_CPP(&te::fusion::BinaryManager::GetBinaryPath,
        void(te::fusion::BinaryManager::*)(const OpBuildTaskPtr &, bool, bool, std::string &) const)
        .stubs()
        .will(invoke(GetBinaryPath_stub));

    BinaryManager::Instance().binaryConfigPathMap_ = {{6, ""}};
    BinaryManager::Instance().ParseAllBinaryInfoConfigPath();

    std::string jsonFilePath;
    res = te::fusion::BinaryManager::Instance().MatchSimplifiedKey(opTask, jsonFilePath);
    EXPECT_EQ(res, true);
    EXPECT_EQ(jsonFilePath, "ascend910/pad_v3_grad/PadV3Grad_6c0a56f859247d7da238b8abd61935c4_high_performance.json");
    res = te::fusion::BinaryManager::Instance().ReuseBinKernelBySimpleKey(opTask);
    EXPECT_EQ(res, true);

    auto opTask1 = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    opTask1->kernel = "te_split_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    opTask1->buildType = FUZZILY_BUILD;
    opTask1->newCompile = true;
    opTask1->kernel = "te_split_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    TbeOpInfo opInfo1("split", "", "Split", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp1 = std::make_shared<TbeOpInfo>(opInfo1);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "float16", "ND", "input0", range1, true, pTbeOp1, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "int32", "ND", "input1", range1, true, pTbeOp1, TT_REQ);
    AddOpParamToTbeOpInfoPtr({9,2,3,4}, "uint32", "ND", "output", range, false, pTbeOp1, TT_DYN);

    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp1);

    (void)pTbeOp1->SetOptions(options);
    pTbeOp1->SetNode(nodePtr);
    pTbeOp1->SetOpImplMode("default");
    res = te::fusion::BinaryManager::Instance().MatchSimplifiedKey(opTask1, jsonFilePath);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, test_MatchSimpleKey3) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("padV3Grad", "PadV3Grad");
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{1,10}};
    AddTensorToOpDesc(true, "input0", {-1,-1}, ge::Format::FORMAT_ND, DT_FLOAT16, range1, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, ge::Format::FORMAT_ND, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(false, "output", {9,2,3,4}, ge::Format::FORMAT_ND, DT_FLOAT16, range, opDescPtr);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opDescPtr, "variable_attr", variableValue);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    teGraphNode.push_back(nodePtr.get());
    std::string opName = teGraphNode[0]->GetName();

    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    opTask->kernel = "te_padV3Grad_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    opTask->buildType = FUZZILY_BUILD;
    opTask->newCompile = true;
    opTask->kernel = "te_padV3Grad_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    TbeOpInfo opInfo("padV3Grad", "", "PadV3Grad", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "float16", "ND", "input0", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "int64", "ND", "input1", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({9,2,3,4}, "float16", "ND", "output", range, false, pTbeOp, TT_REQ);

    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strVal = "reflect";
    TbeAttrValue attr_var("mode", strVal);
    pTbeOp->AddAttrValue(attr_var);

    TbeAttrValue attr_var_1("paddings_contiguous", false);
    pTbeOp->AddAttrValue(attr_var_1);

    std::map<std::string, std::string> options;
    options["deterministic"] = "false";
    (void)pTbeOp->SetOptions(options);
    pTbeOp->SetNode(nodePtr);
    pTbeOp->SetOpImplMode("default");

    MOCKER_CPP(&te::fusion::BinaryManager::GetBinaryPath,
        void(te::fusion::BinaryManager::*)(const OpBuildTaskPtr &, bool, bool, std::string &) const)
        .stubs()
        .will(invoke(GetBinaryPath_stub));

    MOCKER_CPP(&te::fusion::BinaryInfoBase::GetSimpleKeyMode,
        bool(te::fusion::BinaryInfoBase::*)(const std::string &, SimpleKeyModeType &) const)
        .stubs()
        .will(invoke(GetSimpleKeyMode_stub));

    std::string jsonFilePath;
    res = te::fusion::BinaryManager::Instance().MatchSimplifiedKey(opTask, jsonFilePath);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, test_MatchSimpleKey4) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("padV3Grad", "PadV3Grad");
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{1,10}};
    AddTensorToOpDesc(true, "input0", {-1,-1}, ge::Format::FORMAT_ND, DT_FLOAT16, range1, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, ge::Format::FORMAT_ND, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(false, "output", {9,2,3,4}, ge::Format::FORMAT_ND, DT_FLOAT16, range, opDescPtr);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opDescPtr, "variable_attr", variableValue);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    teGraphNode.push_back(nodePtr.get());
    std::string opName = teGraphNode[0]->GetName();

    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    opTask->kernel = "te_padV3Grad_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    opTask->buildType = FUZZILY_BUILD;
    opTask->newCompile = true;
    opTask->kernel = "te_padV3Grad_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    TbeOpInfo opInfo("padV3Grad", "", "PadV3Grad", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "float16", "ND", "input0", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "int64", "ND", "input1", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({9,2,3,4}, "float16", "ND", "output", range, false, pTbeOp, TT_REQ);

    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strVal = "reflect";
    TbeAttrValue attr_var("mode", strVal);
    pTbeOp->AddAttrValue(attr_var);

    TbeAttrValue attr_var_1("paddings_contiguous", false);
    pTbeOp->AddAttrValue(attr_var_1);

    std::map<std::string, std::string> options;
    options["deterministic"] = "false";
    (void)pTbeOp->SetOptions(options);
    pTbeOp->SetNode(nodePtr);
    pTbeOp->SetOpImplMode("default");

    MOCKER_CPP(&te::fusion::BinaryManager::GetBinaryPath,
        void(te::fusion::BinaryManager::*)(const OpBuildTaskPtr &, bool, bool, std::string &) const)
        .stubs()
        .will(invoke(GetBinaryPath_stub));

    MOCKER_CPP(&te::fusion::BinaryInfoBase::GetSimpleKeyMode,
        bool(te::fusion::BinaryInfoBase::*)(const std::string &, SimpleKeyModeType &) const)
        .stubs()
        .will(invoke(GetSimpleKeyMode_stub));

    std::string jsonFilePath;
    res = te::fusion::BinaryManager::Instance().MatchSimplifiedKey(opTask, jsonFilePath);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, test_SetBinaryConstValue) {
    TeFusionManager *pInstance = te::fusion::TeFusionManager::GetInstance();

    TbeOpTensor tensor("x", {3}, "float32", "ND");

    // const value isn't null
    std::vector<float> const_value1;
    const_value1.push_back(1.0);
    const_value1.push_back(std::numeric_limits<float>::quiet_NaN());
    const_value1.push_back(std::numeric_limits<float>::infinity());
    const_value1.push_back(-std::numeric_limits<float>::infinity());

    TbeAttrValue attrValue1("x", const_value1);
    tensor.SetConstValue(attrValue1);

    nlohmann::json curDynamicInfo;
    te::fusion::ShapeGeneralization::SetBinaryConstValue(tensor, curDynamicInfo);
    printf("jsonRes is %s\n", curDynamicInfo.dump().c_str());
    EXPECT_EQ(curDynamicInfo.contains("const_value_null_desc"), true);
    EXPECT_EQ(curDynamicInfo["const_value_null_desc"][1], "nan");
    EXPECT_EQ(curDynamicInfo["const_value_null_desc"][2], "inf");
    EXPECT_EQ(curDynamicInfo["const_value_null_desc"][3], "-inf");

    // const value is list null
    std::vector<float> const_value2;
    const_value2.push_back(1.0);
    TbeAttrValue attrValue2("x", const_value2);
    tensor.SetConstValue(attrValue2);
    nlohmann::json curDynamicInfo2;
    te::fusion::ShapeGeneralization::SetBinaryConstValue(tensor, curDynamicInfo2);
    printf("jsonRes is %s\n", curDynamicInfo2.dump().c_str());
    EXPECT_EQ(curDynamicInfo2.contains("const_value_null_desc"), false);
    
    // const value is list null
    TbeAttrValue attrValue3("x", 2.0);
    tensor.SetConstValue(attrValue3);
    nlohmann::json curDynamicInfo3;
    te::fusion::ShapeGeneralization::SetBinaryConstValue(tensor, curDynamicInfo3);
    printf("jsonRes is %s\n", curDynamicInfo3.dump().c_str());
    EXPECT_EQ(curDynamicInfo3.contains("const_value_null_desc"), false);
}

TEST(TeFusionBinaryUTest, test_GenerateNormalizeFusionAttrTmpJson) {
    TeFusionManager *pInstance = te::fusion::TeFusionManager::GetInstance();

    TbeOpInfo op_info1("Range", "", "Range", "AIcoreEngine");
    TbeOpInfoPtr op_info1_ptr = std::make_shared<TbeOpInfo>(op_info1);
    std::vector<float> const_value1;
    const_value1.push_back(std::numeric_limits<float>::quiet_NaN());
    const_value1.push_back(std::numeric_limits<float>::infinity());
    const_value1.push_back(-std::numeric_limits<float>::infinity());
    std::vector<float> const_value2;
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "start", const_value1, true, op_info1);
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "limit", const_value1, true, op_info1);
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "delta", const_value1, true, op_info1);
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "delta", const_value2, false, op_info1);

    // const value is null
    TbeOpInfo op_info2("Range", "", "Range", "AIcoreEngine");
    TbeOpInfoPtr op_info2_ptr = std::make_shared<TbeOpInfo>(op_info2);
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "start", const_value2, true, op_info2);
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "limit", const_value2, true, op_info2);
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "delta", const_value2, true, op_info2);
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "delta", const_value2, false, op_info2);
}

TEST(TeFusionBinaryUTest, test_MatchSimpleKey5) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("mul", "Mul");
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{1,10}};
    AddTensorToOpDesc(true, "input0", {-1,-1}, FORMAT_NCHW, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, FORMAT_NCHW, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(false, "output", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opDescPtr, "variable_attr", variableValue);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    teGraphNode.push_back(nodePtr.get());
    std::string opName = teGraphNode[0]->GetName();

    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    opTask->kernel = "te_mul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    opTask->buildType = FUZZILY_BUILD;
    opTask->newCompile = true;
    opTask->kernel = "te_mul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    TbeOpInfo opInfo("mul", "", "Mul", "AIcoreEngine");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opInfo);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "int64", "NCHW", "input0", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({-1,-1}, "int64", "NCHW", "input1", range1, true, pTbeOp, TT_REQ);
    AddOpParamToTbeOpInfoPtr({9,2,3,4}, "int64", "NHWC", "output", range, false, pTbeOp, TT_REQ);

    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    TbeAttrValue attr_var("_input_name_key", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    pTbeOp->AddAttrValue(attr_var);
    TbeAttrValue attr_var_1("_dst_type", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    pTbeOp->AddAttrValue(attr_var_1);
    pTbeOp->SetNode(nodePtr);

    std::map<std::string, std::string> options;
    options["deterministic"] = "true";
    options["ge.opDebugLevel"] = "1";
    (void)pTbeOp->SetOptions(options);
    pTbeOp->SetOpImplMode("default");

    TeFusionManager *pInstance = te::fusion::TeFusionManager::GetInstance();
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    std::string jsonFilePath = te::fusion::RealPath(currentFilePath) + "/" +
        "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    res = BinaryManager::Instance().SetBinaryReuseResult(opTask, jsonFilePath);

    jsonFilePath = "testfile";
    res = BinaryManager::Instance().SetOmBinaryReuseResult(opTask, jsonFilePath);
    EXPECT_EQ(res, false);

    nlohmann::json binaryListJson;
    res = BinaryManager::Instance().GetJsonValueByJsonFilePath(jsonFilePath, binaryListJson);
    EXPECT_EQ(res, false);
}

TEST(TeFusionBinaryUTest, test_opp_latest_env) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase";
    setenv("ASCEND_OPP_PATH", currentFilePath.c_str(), 1);
    setenv("ASCEND_HOME_PATH", currentFilePath.c_str(), 1);
    TeConfigInfo::Instance().InitEnvItem();
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::ShortSocVersion)] = "ascend910";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::CoreType)] = "ascend910";
    const std::map<std::string, std::string> options = {
        {"op.binary.builtin", "6|/built-in/op_impl/ai_core/tbe/kernel/"},
        {"op.binary.custom", "100|/vendors/customize/op_impl/ai_core/tbe/kernel/, 101|/vendors/mdc/op_impl/ai_core/tbe/kernel/"}
    };
    // get binary file path
    te::fusion::BinaryManager::Instance().Finalize();
    te::fusion::BinaryManager::Instance().Initialize(options);
    auto binInfoPtr = te::fusion::BinaryManager::Instance().binaryInfoPtrMap_[6];
    // EXPECT_EQ(binInfoPtr->GetOppBinFlag(), true);
}

TEST(TeFusionBinaryUTest, test_generate_dtype_format_mode) {
    nlohmann::json binary_info_params;
    binary_info_params["Sub"]["inputs"]["dtypeMode"] = "";
    binary_info_params["Sub"]["inputs"]["formatMode"] = "";
    binary_info_params["Add"]["outputs"]["dtypeMode"] = "normal";
    binary_info_params["Add"]["outputs"]["formatMode"] = "normal";
    std::string opType = "Add";
    BinaryInfoBasePtr binaryInfo = nullptr;
    TE_FUSION_MAKE_SHARED(binaryInfo = std::make_shared<BinaryInfoBase>(), return);
    const auto inputOutputIter = binary_info_params.find(opType);
    const json &inputOutputJson = inputOutputIter.value();
    DtypeFormatMode inputMode;
    DtypeFormatMode outputMode;
    binaryInfo->outputMode_["Add"] = outputMode;


    binaryInfo->GenerateDtypeFormatMode(opType, binary_info_params);
    binaryInfo->inputMode_["Add"] = inputMode;
    binaryInfo->GenerateDtypeFormatMode(opType, binary_info_params);
}

TEST(TeFusionBinaryUTest, test_generate_simple_key_list_001) {
    std::string opType = "Abs";
    nlohmann::json binaryListJson;
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/binary_list.json";

    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, binaryListJson);
    nlohmann::json binaryList = binaryListJson["binaryList"];
    std::unordered_map<std::string, std::string> simpleKeyInfoMap;
    BinaryInfoBasePtr binaryInfo = nullptr;
    binaryInfo->GenerateSimpleKeyList(opType, binaryList, simpleKeyInfoMap);
}

TEST(TeFusionBinaryUTest, test_generate_simple_key_list_002) {
    std::string opType = "Abs";
    nlohmann::json binaryListJson;
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/binary_list_no_key.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, binaryListJson);
    nlohmann::json binaryList = binaryListJson["binaryList"];
    std::unordered_map<std::string, std::string> simpleKeyInfoMap;
    BinaryInfoBasePtr binaryInfo = nullptr;
    binaryInfo->GenerateSimpleKeyList(opType, binaryList, simpleKeyInfoMap);
}

TEST(TeFusionBinaryUTest, test_generate_simple_key_list_003) {
    std::string opType = "Abs";
    nlohmann::json binaryListJson;
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/binary_list_no_key.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, binaryListJson);
    nlohmann::json binaryList = binaryListJson["binaryList"];
    std::unordered_map<std::string, std::string> simpleKeyInfoMap;
    BinaryInfoBasePtr binaryInfo = nullptr;
    binaryInfo->GenerateSimpleKeyList(opType, binaryList, simpleKeyInfoMap);
}

TEST(TeFusionBinaryUTest, test_generate_simple_key_list_004) {
    std::string opType = "Abs";
    nlohmann::json binaryListJson;
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub/binary_list_path_null.json";
    std::string path = te::fusion::RealPath(currentFilePath);
    te::fusion::TeFileUtils::GetJsonValueFromJsonFile(path, binaryListJson);
    nlohmann::json binaryList = binaryListJson["binaryList"];
    std::unordered_map<std::string, std::string> simpleKeyInfoMap;
    BinaryInfoBasePtr binaryInfo = nullptr;
    binaryInfo->GenerateSimpleKeyList(opType, binaryList, simpleKeyInfoMap);
}
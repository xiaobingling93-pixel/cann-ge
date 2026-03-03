/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include <dlfcn.h>
#include <assert.h>
#include <fcntl.h>
#include "securec.h"
#include <stdio.h>
#include "dvpp_rt_kernel.h"

#define private public  // hack complier
#define protected public


#undef private
#undef protected

using namespace std;

constexpr uint32_t SQE_FIELD_LEN = 160;

struct Sqe {
    uint32_t field[SQE_FIELD_LEN];
    void* args{nullptr};
    uint32_t argsSize{0};
};

Sqe mockSqe;

class DvppRtKernelUt : public testing::Test {
    public:
        DvppRtKernelUt() {}

    protected:
        virtual void SetUp() {}

        virtual void TearDown() {
            GlobalMockObject::verify();
            GlobalMockObject::reset();
        }
};

TEST_F(DvppRtKernelUt, create_generate_dvpp_sqe_outputs_001)
{
    gert::KernelContext context{};
    const ge::FastNode* node = new ge::FastNode();
    gert::Chain chain;
    chain.any_value_.deleter = nullptr;
    MOCKER_CPP(&gert::KernelContext::GetOutput).stubs().will(returnValue(&chain));
    ge::Status ret = gert::kernel::CreateGenerateDvppSqeOutputs(node, &context);
    delete node;
    EXPECT_EQ(ge::SUCCESS, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, generate_sqe_and_launch_task_001)
{
    gert::KernelContext context{};
    gert::ContinuousVector conVec{};
    gert::ContinuousVector* ptrVec = &conVec;
    string str = "test";

    MOCKER_CPP(&gert::KernelContext::GetInputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeInputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeOutputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    ge::Status ret = gert::kernel::GenerateSqeAndLaunchTask(&context);

    EXPECT_EQ(ge::FAILED, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, generate_sqe_and_launch_task_002)
{
    gert::KernelContext context{};
    gert::ContinuousVector conVec{};
    gert::ContinuousVector* ptrVec = &conVec;
    string str = "test";
    uint32_t *mockSqePtr = (uint32_t *)&mockSqe;

    MOCKER_CPP(&gert::KernelContext::GetInputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<uint32_t*>).stubs().will(returnValue(&mockSqePtr));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeInputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeOutputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    MOCKER_CPP(gert::kernel::BuildDvppCmdlistV2).stubs().will(returnValue(-1));
    MOCKER_CPP(gert::kernel::FreeSqeReadOnlyMem).stubs();
    MOCKER_CPP(memset_s).stubs().will(returnValue(0));
    ge::Status ret = gert::kernel::GenerateSqeAndLaunchTask(&context);
    EXPECT_EQ(ge::FAILED, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, generate_sqe_and_launch_task_003)
{
    gert::KernelContext context{};
    gert::ContinuousVector conVec{};
    gert::ContinuousVector* ptrVec = &conVec;
    string str = "test";
    uint32_t *mockSqePtr = (uint32_t *)&mockSqe;

    MOCKER_CPP(&gert::KernelContext::GetInputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<uint32_t*>).stubs().will(returnValue(&mockSqePtr));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeInputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeOutputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    MOCKER_CPP(gert::kernel::BuildDvppCmdlistV2).stubs().will(returnValue(0));
    MOCKER_CPP(gert::kernel::StarsBatchTaskLaunch).stubs().will(returnValue(0));
    MOCKER_CPP(memset_s).stubs().will(returnValue(0));
    ge::Status ret = gert::kernel::GenerateSqeAndLaunchTask(&context);
    EXPECT_EQ(ge::SUCCESS, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, generate_sqe_and_launch_task_004)
{
    gert::KernelContext context{};
    gert::ContinuousVector conVec{};
    gert::ContinuousVector* ptrVec = &conVec;
    string str = "DecodeAndCropJpeg";
    uint32_t *mockSqePtr = (uint32_t *)&mockSqe;
    gert::TensorData* pTensor = reinterpret_cast<gert::TensorData *>(mockSqePtr);

    MOCKER_CPP(&gert::KernelContext::GetInputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<uint32_t*>).stubs().will(returnValue(&mockSqePtr));
    MOCKER_CPP(&gert::KernelContext::GetInputValue<gert::TensorData*>).stubs().will(returnValue(pTensor));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeInputNum).stubs().will(returnValue(1U));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeOutputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(gert::kernel::BuildDvppCmdlistV2).stubs().will(returnValue(0));
    MOCKER_CPP(gert::kernel::StarsMultipleTaskLaunch).stubs().will(returnValue(0));
    MOCKER_CPP(memset_s).stubs().will(returnValue(0));
    ge::Status ret = gert::kernel::GenerateSqeAndLaunchTask(&context);
    EXPECT_EQ(ge::SUCCESS, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, generate_sqe_and_launch_task_005)
{
    gert::KernelContext context{};
    uint32_t tmp[32] = {1};
    uint32_t* tmpPtr = tmp;
    gert::ContinuousVector conVec{};
    conVec.SetSize(1);
    gert::ContinuousVector* ptrVec = &conVec;
    string str = "test";
    uint32_t *mockSqePtr = (uint32_t *)&mockSqe;

    MOCKER_CPP(&gert::KernelContext::GetInputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<uint32_t*>).stubs().will(returnValue(&mockSqePtr));
    MOCKER_CPP(&gert::GertTensorData::GetAddr).stubs().will(returnValue((void*)tmpPtr));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    MOCKER_CPP(gert::kernel::BuildDvppCmdlistV2).stubs().will(returnValue(0));
    MOCKER_CPP(gert::kernel::StarsBatchTaskLaunch).stubs().will(returnValue(0));
    MOCKER_CPP(memset_s).stubs().will(returnValue(0));
    ge::Status ret = gert::kernel::GenerateSqeAndLaunchTask(&context);
    EXPECT_EQ(ge::SUCCESS, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, generate_sqe_and_launch_task_006)
{
    gert::KernelContext context{};
    gert::ContinuousVector conVec{};
    gert::ContinuousVector* ptrVec = &conVec;
    string str = "test";
    uint32_t *mockSqePtr = (uint32_t *)&mockSqe;

    MOCKER_CPP(&gert::KernelContext::GetInputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<uint32_t*>).stubs().will(returnValue(&mockSqePtr));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeInputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(&gert::DvppContext::GetComputeNodeOutputNum).stubs().will(returnValue(0U));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    MOCKER_CPP(gert::kernel::BuildDvppCmdlistV2).stubs().will(returnValue(0));
    MOCKER_CPP(gert::kernel::StarsBatchTaskLaunch).stubs().will(returnValue(-1));
    MOCKER_CPP(memset_s).stubs().will(returnValue(0));

    ge::Status ret = gert::kernel::GenerateSqeAndLaunchTask(&context);
    EXPECT_EQ(ge::FAILED, ret);

    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, CreateCalcWorkspaceSizeOutputs_001)
{
    gert::KernelContext context{};
    ge::Status ret = gert::kernel::CreateCalcWorkspaceSizeOutputs(nullptr, &context);
    EXPECT_EQ(ge::FAILED, ret);

    ret = gert::kernel::CreateCalcWorkspaceSizeOutputs(nullptr, nullptr);
    EXPECT_EQ(ge::FAILED, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, CalcOpWorkSpaceSize_001)
{
    gert::KernelContext context{};
    gert::ContinuousVector conVec{};
    conVec.SetSize(0);
    gert::ContinuousVector* ptrVec = &conVec;
    string str = "AdjustBrightness";

    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    MOCKER_CPP(gert::kernel::CalcOpWorkspaceMemSize).stubs().will(returnValue(0));
    ge::Status ret = gert::kernel::CalcOpWorkSpaceSize(&context);
    EXPECT_EQ(ge::SUCCESS, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, CalcOpWorkSpaceSize_002)
{
    gert::KernelContext context{};
    gert::ContinuousVector conVec{};
    conVec.SetSize(1);
    gert::ContinuousVector* ptrVec = &conVec;
    size_t tmpData[10] = {0};
    string str = "test";

    MOCKER_CPP(&gert::ContinuousVector::MutableData).stubs().will(returnValue((void*)tmpData));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    MOCKER_CPP(gert::kernel::CalcOpWorkspaceMemSize).stubs().will(returnValue(0));
    ge::Status ret = gert::kernel::CalcOpWorkSpaceSize(&context);
    EXPECT_EQ(ge::SUCCESS, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, CalcOpWorkSpaceSize_003)
{
    gert::KernelContext context{};
    gert::ContinuousVector conVec{};
    conVec.SetSize(1);
    gert::ContinuousVector* ptrVec = &conVec;
    size_t tmpData[10] = {0};
    string str = "test_node";

    MOCKER_CPP(&gert::ContinuousVector::MutableData).stubs().will(returnValue((void*)tmpData));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<gert::ContinuousVector>).stubs().will(returnValue(ptrVec));
    MOCKER_CPP(gert::kernel::CalcOpWorkspaceMemSize).stubs().will(returnValue(-1));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    ge::Status ret = gert::kernel::CalcOpWorkSpaceSize(&context);
    EXPECT_EQ(ge::FAILED, ret);
    GlobalMockObject::reset();

    MOCKER_CPP(&gert::ContinuousVector::MutableData).stubs().will(returnValue((void*)tmpData));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<gert::ContinuousVector>).stubs().will(returnValue(static_cast<gert::ContinuousVector*>(nullptr)));
    MOCKER_CPP(gert::kernel::CalcOpWorkspaceMemSize).stubs().will(returnValue(2));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    ret = gert::kernel::CalcOpWorkSpaceSize(&context);
    EXPECT_EQ(ge::FAILED, ret);
    GlobalMockObject::reset();

    MOCKER_CPP(&gert::ContinuousVector::MutableData).stubs().will(returnValue((void*)tmpData));
    MOCKER_CPP(&gert::KernelContext::GetOutputPointer<gert::ContinuousVector>).stubs().will(returnValue(static_cast<gert::ContinuousVector*>(ptrVec)));
    MOCKER_CPP(gert::kernel::CalcOpWorkspaceMemSize).stubs().will(returnValue(2));
    MOCKER_CPP(&gert::DvppContext::GetNodeType).stubs().will(returnValue(str));
    ret = gert::kernel::CalcOpWorkSpaceSize(&context);
    EXPECT_EQ(ge::FAILED, ret);
    GlobalMockObject::reset();
}

TEST_F(DvppRtKernelUt, interface_test_001)
{
    MOCKER_CPP(dlopen).stubs().will(returnValue(static_cast<void *>(nullptr)));
    MOCKER_CPP(dlsym).stubs().will(returnValue(static_cast<void *>(nullptr)));
    gert::DvppContext context{};
    std::vector<void*> ioAddrs{};
    DvppSqeInfo sqeInfo{};
    size_t memSize = 1;
    void *taskSqe = nullptr;
    void *stream = nullptr;
    
    ge::Status ret = gert::kernel::BuildDvppCmdlistV2(&context, ioAddrs, sqeInfo);
    EXPECT_EQ(ge::FAILED, ret);
    ret = gert::kernel::BuildDvppCmdlistV2(&context, ioAddrs, sqeInfo);
    EXPECT_EQ(ge::FAILED, ret);
    ret = gert::kernel::CalcOpWorkspaceMemSize(&context, memSize);
    EXPECT_EQ(ge::FAILED, ret);
    ret = gert::kernel::StarsBatchTaskLaunch(taskSqe, 0U, stream);
    EXPECT_EQ(ge::FAILED, ret);
    ret = gert::kernel::StarsMultipleTaskLaunch(taskSqe, 0U, stream);
    EXPECT_EQ(ge::FAILED, ret);
    gert::kernel::FreeSqeReadOnlyMem(taskSqe);

    GlobalMockObject::reset();
}

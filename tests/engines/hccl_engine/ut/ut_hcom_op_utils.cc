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
#include "hcom_op_utils.h"
#include "mocker/mocker.h"
#include "ge/op_desc.h"
#include "ge/tensor_desc.h"
#include "ge/tensor_utils.h"
#include "sal/sal_interface.h"

using namespace hccl;

class HcomOpUtilsTest : public testing::Test {
protected:
  // 重置所有mock对象，确保测试用例之间的独立性
  void SetUp() override {
    MOCKER_RESET_ALL();
  }

  // 本测试类无需特殊清理，保持空实现
  void TearDown() override {
  }

  // 辅助函数：设置SalGetDataTypeSize的mock行为
  void SetUpMockSalGetDataTypeSize(HcclDataType dataType, u32 size) {
    MOCKER(SalGetDataTypeSize).ExpectCall(
        With(dataType),
        WithOut(size))
        .WillOnce(Return(HCCL_SUCCESS));
  }

  // 辅助函数：设置TensorUtils::GetSize的mock行为
  void SetUpMockTensorUtilsGetSize(int64_t tensorSize) {
    MOCKER(ge::TensorUtils::GetSize).ExpectCall(
        WithAny(),
        WithOut(tensorSize))
        .WillOnce(Return(ge::GRAPH_SUCCESS));
  }

  class MockOpDesc : public ge::OpDesc {
  public:
    MOCK_METHOD(u64, GetInputsSize, (), (const, override));
    MOCK_METHOD(const ge::TensorDesc*, GetInputDescPtr, (u64), (const, override));
  };

  class MockTensorDesc : public ge::TensorDesc {
  public:
    MOCK_METHOD(const ge::Shape&, GetShape, (), (const, override));
  };

  // 模拟Shape类，用于控制张量形状信息
  class MockShape : public ge::Shape {
  public:
    MOCK_METHOD(int64_t, GetShapeSize, (), (const, override));
  };
};

// 测试用例：ALLREDUCE算子正常路径测试
// 测试场景：使用有效的输入参数调用GetAccuracyCountFromOpDesc函数，算子类型为ALLREDUCE
// 预期行为：函数返回HCCL_SUCCESS，并且计算得到正确的count值
// 测试目的：验证ALLREDUCE算子的count计算逻辑是否正确
TEST_F(HcomOpUtilsTest, Ut_GetAccuracyCountFromOpDesc_When_AllReduceWithValidInput_Expect_Success) {
  // 准备测试环境：创建mock对象并设置期望行为
  auto op = std::make_shared<MockOpDesc>();
  EXPECT_CALL(*op, GetInputsSize()).WillOnce(Return(1));
  
  // 创建mock张量描述和形状对象
  auto tensorDesc = std::make_shared<MockTensorDesc>();
  auto shape = std::make_shared<MockShape>();
  // 设置形状大小为256（元素数量）
  EXPECT_CALL(*shape, GetShapeSize()).WillOnce(Return(256));
  EXPECT_CALL(*tensorDesc, GetShape()).WillOnce(Return(*shape));
  EXPECT_CALL(*op, GetInputDescPtr(0)).WillOnce(Return(tensorDesc.get()));

  SetUpMockSalGetDataTypeSize(HCCL_DATA_TYPE_FP32, 4);
  SetUpMockTensorUtilsGetSize(1024);
  
  // 测试参数设置
  std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_ALLREDUCE;  // 算子类型为ALLREDUCE
  HcclDataType dataType = HCCL_DATA_TYPE_FP32;                 // 数据类型为float32
  u64 count = 0;                                               // 输出参数：count值
  u32 rankSize = 1;                                            // rank大小为1

  // 执行被测函数
  HcclResult ret = HcomOpUtils::GetAccuracyCountFromOpDesc(op, sCollectiveType, dataType, count, rankSize);

  // 验证结果：函数应返回成功，且count值应为256（1024字节 / 4字节/元素）
  EXPECT_EQ(ret, HCCL_SUCCESS) << "Expected HCCL_SUCCESS, got " << ret;
  EXPECT_EQ(count, 256) << "Expected count 256, got " << count;
}

// 测试用例：BROADCAST算子正常路径测试
// 测试场景：使用有效的输入参数调用GetAccuracyCountFromOpDesc函数，算子类型为BROADCAST
// 预期行为：函数返回HCCL_SUCCESS，并且计算得到正确的count值（包含对齐处理）
// 测试目的：验证BROADCAST算子的count计算逻辑（包括512字节对齐）是否正确
TEST_F(HcomOpUtilsTest, Ut_GetAccuracyCountFromOpDesc_When_BroadcastWithValidInput_Expect_Success) {
  // 准备测试环境：创建mock对象并设置期望行为
  auto op = std::make_shared<MockOpDesc>();
  EXPECT_CALL(*op, GetInputsSize()).WillOnce(Return(1));
  
  // 创建mock张量描述和形状对象
  auto tensorDesc = std::make_shared<MockTensorDesc>();
  auto shape = std::make_shared<MockShape>();
  // 设置形状大小为100（元素数量）
  EXPECT_CALL(*shape, GetShapeSize()).WillOnce(Return(100));
  EXPECT_CALL(*tensorDesc, GetShape()).WillOnce(Return(*shape));
  EXPECT_CALL(*op, GetInputDescPtr(0)).WillOnce(Return(tensorDesc.get()));

  SetUpMockSalGetDataTypeSize(HCCL_DATA_TYPE_FP32, 4);
  
  // 测试参数设置
  std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_BROADCAST;  // 算子类型为BROADCAST
  HcclDataType dataType = HCCL_DATA_TYPE_FP32;                  // 数据类型为float32
  u64 count = 0;                                                // 输出参数：count值
  u32 rankSize = 4;                                             // rank大小为4

  // 执行被测函数
  HcclResult ret = HcomOpUtils::GetAccuracyCountFromOpDesc(op, sCollectiveType, dataType, count, rankSize);

  // 验证结果：函数应返回成功，且count值应为128
  // 计算过程：inputSize = 100 * 4 = 400字节，对齐到512字节，512 / 4 = 128元素
  EXPECT_EQ(ret, HCCL_SUCCESS) << "Expected HCCL_SUCCESS, got " << ret;
  EXPECT_EQ(count, 128) << "Expected count 128, got " << count;
}

// 测试用例：RECEIVE算子错误路径测试
// 测试场景：使用RECEIVE算子类型调用GetAccuracyCountFromOpDesc函数
// 预期行为：函数返回HCCL_E_PARA错误，因为RECEIVE算子不支持获取count
// 测试目的：验证函数对不支持算子类型的错误处理逻辑是否正确
TEST_F(HcomOpUtilsTest, Ut_GetAccuracyCountFromOpDesc_When_ReceiveOp_Expect_E_PARA) {
  // 准备测试环境：创建mock对象并设置期望行为
  auto op = std::make_shared<MockOpDesc>();
  // 设置算子输入数量为1
  EXPECT_CALL(*op, GetInputsSize()).WillOnce(Return(1));
  
  // 创建mock张量描述和形状对象
  auto tensorDesc = std::make_shared<MockTensorDesc>();
  auto shape = std::make_shared<MockShape>();
  // 设置形状大小为100（元素数量）
  EXPECT_CALL(*shape, GetShapeSize()).WillOnce(Return(100));
  EXPECT_CALL(*tensorDesc, GetShape()).WillOnce(Return(*shape));
  EXPECT_CALL(*op, GetInputDescPtr(0)).WillOnce(Return(tensorDesc.get()));
  
  // 设置SalGetDataTypeSize的mock行为，返回float32的大小为4字节
  SetUpMockSalGetDataTypeSize(HCCL_DATA_TYPE_FP32, 4);
  
  // 测试参数设置：使用RECEIVE算子类型（不支持）
  std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_RECEIVE;  // 算子类型为RECEIVE
  HcclDataType dataType = HCCL_DATA_TYPE_FP32;                // 数据类型为float32
  u64 count = 0;                                              // 输出参数：count值
  u32 rankSize = 4;                                           // rank大小为4

  // 执行被测函数
  HcclResult ret = HcomOpUtils::GetAccuracyCountFromOpDesc(op, sCollectiveType, dataType, count, rankSize);

  // 验证结果：函数应返回HCCL_E_PARA错误，因为RECEIVE算子不支持获取count
  EXPECT_EQ(ret, HCCL_E_PARA) << "Expected HCCL_E_PARA for RECEIVE op, got " << ret;
}

// 测试用例：数据类型大小为零的参数验证测试
// 测试场景：调用GetAccuracyCountFromOpDesc函数时，数据类型大小为零
// 预期行为：函数返回HCCL_E_PARA错误，因为数据类型大小不能为零
// 测试目的：验证函数对无效数据类型大小的参数验证逻辑是否正确
TEST_F(HcomOpUtilsTest, Ut_GetAccuracyCountFromOpDesc_When_DataTypeSizeZero_Expect_E_PARA) {
  // 准备测试环境：创建mock对象并设置期望行为
  auto op = std::make_shared<MockOpDesc>();
  EXPECT_CALL(*op, GetInputsSize()).WillOnce(Return(1));
  
  // 创建mock张量描述和形状对象
  auto tensorDesc = std::make_shared<MockTensorDesc>();
  auto shape = std::make_shared<MockShape>();
  // 设置形状大小为100（元素数量）
  EXPECT_CALL(*shape, GetShapeSize()).WillOnce(Return(100));
  EXPECT_CALL(*tensorDesc, GetShape()).WillOnce(Return(*shape));
  EXPECT_CALL(*op, GetInputDescPtr(0)).WillOnce(Return(tensorDesc.get()));
  
  // 设置SalGetDataTypeSize的mock行为，返回数据类型大小为零（无效值）
  SetUpMockSalGetDataTypeSize(HCCL_DATA_TYPE_FP32, 0);
  
  // 测试参数设置
  std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_ALLGATHER;  // 算子类型为ALLGATHER
  HcclDataType dataType = HCCL_DATA_TYPE_FP32;                  // 数据类型为float32
  u64 count = 0;                                                // 输出参数：count值
  u32 rankSize = 4;                                             // rank大小为4

  // 执行被测函数
  HcclResult ret = HcomOpUtils::GetAccuracyCountFromOpDesc(op, sCollectiveType, dataType, count, rankSize);

  // 验证结果：函数应返回HCCL_E_PARA错误，因为数据类型大小不能为零
  EXPECT_EQ(ret, HCCL_E_PARA) << "Expected HCCL_E_PARA for zero dataTypeSize, got " << ret;
}

// 测试用例：rankSize为零的边界条件测试
// 测试场景：调用GetAccuracyCountFromOpDesc函数时，rankSize参数为零
// 预期行为：函数返回HCCL_E_PARA错误，因为rankSize不能为零（会导致除零错误）
// 测试目的：验证函数对无效rankSize参数的边界条件处理逻辑是否正确
TEST_F(HcomOpUtilsTest, Ut_GetAccuracyCountFromOpDesc_When_RankSizeZero_Expect_E_PARA) {
  // 准备测试环境：创建mock对象并设置期望行为
  auto op = std::make_shared<MockOpDesc>();
  EXPECT_CALL(*op, GetInputsSize()).WillOnce(Return(1));
  
  // 创建mock张量描述和形状对象
  auto tensorDesc = std::make_shared<MockTensorDesc>();
  auto shape = std::make_shared<MockShape>();
  // 设置形状大小为100（元素数量）
  EXPECT_CALL(*shape, GetShapeSize()).WillOnce(Return(100));
  EXPECT_CALL(*tensorDesc, GetShape()).WillOnce(Return(*shape));
  EXPECT_CALL(*op, GetInputDescPtr(0)).WillOnce(Return(tensorDesc.get()));
  
  // 设置SalGetDataTypeSize的mock行为，返回float32的大小为4字节
  SetUpMockSalGetDataTypeSize(HCCL_DATA_TYPE_FP32, 4);
  
  // 测试参数设置：使用REDUCESCATTER算子类型，rankSize设置为零（无效值）
  std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;  // 算子类型为REDUCESCATTER
  HcclDataType dataType = HCCL_DATA_TYPE_FP32;                      // 数据类型为float32
  u64 count = 0;                                                    // 输出参数：count值
  u32 rankSize = 0;                                                 // rank大小为零（无效值）

  // 执行被测函数
  HcclResult ret = HcomOpUtils::GetAccuracyCountFromOpDesc(op, sCollectiveType, dataType, count, rankSize);

  // 验证结果：函数应返回HCCL_E_PARA错误，因为rankSize不能为零
  EXPECT_EQ(ret, HCCL_E_PARA) << "Expected HCCL_E_PARA for zero rankSize, got " << ret;
}
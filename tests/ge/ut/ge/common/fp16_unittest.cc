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

#include "common/fp16_t/fp16_t.h"

namespace ge {
namespace formats {
class UtestFP16 : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFP16, fp16_to_other) {
  fp16_t test;
  float num = test.ToFloat();
  EXPECT_EQ(num, 0.0);

  double num2 = test.ToDouble();
  EXPECT_EQ(num2, 0);

  int16_t num3 = test.ToInt16();
  EXPECT_EQ(num3, 0);

  int32_t num4 = test.ToInt32();
  EXPECT_EQ(num4, 0);

  int8_t num5 = test.ToInt8();
  EXPECT_EQ(num5, 0);

  uint16_t num6 = test.ToUInt16();
  EXPECT_EQ(num6, 0);

  uint32_t num7 = test.ToUInt16();
  EXPECT_EQ(num7, 0);

  uint8_t num8 = test.ToUInt8();
  EXPECT_EQ(num8, 0);

  int32_t num9 = test.ToUInt32();
  EXPECT_EQ(num9, 0);
}

TEST_F(UtestFP16, OperatorAdd_success) {
  fp16_t test1(1);
  fp16_t test2(1);
  test1 = test1 + test2;
  EXPECT_EQ(test1.val, 2);
}

TEST_F(UtestFP16, OperatorSubtract_success) {
  fp16_t test1(1);
  fp16_t test2(1);
  test1 = test1 - test2;
  EXPECT_EQ(test1.val, 0);
}

TEST_F(UtestFP16, OperatorMultiply_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  test1 = test1 * test2;
  EXPECT_EQ(test1.val, 0);
}

TEST_F(UtestFP16, OperatorEqual_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  fp16_t test3(1);
  EXPECT_EQ(test1 == test2, false);
  EXPECT_EQ(test1 == test3, true);
}

TEST_F(UtestFP16, OperatorGreaterThan_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  fp16_t test3(1);
  EXPECT_EQ(test1 > test2, false);
  EXPECT_EQ(test2 > test1, true);
  fp16_t test4((uint16_t)200000U);
  fp16_t test5((uint16_t)300000U);
  EXPECT_EQ(test4 > test5, true);
  EXPECT_EQ(test5 > test4, false);
}

TEST_F(UtestFP16, OperatorEqualOrGreaterThan_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  fp16_t test3(1);
  EXPECT_EQ(test1 >= test2, false);
  EXPECT_EQ(test1 >= test3, true);
}

TEST_F(UtestFP16, OperatorEqualOrLessThan_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  fp16_t test3(3);
  EXPECT_EQ(test1 <= test2, true);
  EXPECT_EQ(test3 <= test2, false);
  fp16_t test4((uint16_t)200000U);
  fp16_t test5((uint16_t)300000U);
  EXPECT_EQ(test4 <= test5, false);
  EXPECT_EQ(test5 <= test4, true);
}

TEST_F(UtestFP16, OperatorEqualToTagFp16_success) {
  fp16_t test(1);
  fp16_t test1(1);
  test1 = test;
  EXPECT_EQ(test1.val, 1);

  float32_t val_f32 = 4294967296;
  test1 = val_f32;
  EXPECT_EQ(test1.val, 31743);

  float64_t val_f64= 8589934592;
  test1 = val_f64;
  EXPECT_EQ(test1.val, 31743);

  fp16_t test4((uint16_t)kFp64ExpMask);
  test1 = test4;
  EXPECT_EQ(test1.val, 0);

  fp16_t test5(0);
  test1 = test5;
  EXPECT_EQ(test1.val, 0);
}

TEST_F(UtestFP16, OperatorEqualTofloat32_t_success) {
  fp16_t test1(1);
  float32_t test2= 2.0F;
  test1 = test2;
  EXPECT_EQ(test1.val, 16384);
}

TEST_F(UtestFP16, OperatorEqualTofloat64_t_success) {
  fp16_t test1(1);
  float64_t test2= 2.0F;
  test1 = test2;
  EXPECT_EQ(test1.val, 16384);
}

TEST_F(UtestFP16, OperatorEqualToint32_t_success) {
  fp16_t test1(1);
  int32_t test2= 2;
  test1 = test2;
  EXPECT_EQ(test1.val, 16384);
}

TEST_F(UtestFP16, Float32_t_success) {
  fp16_t test(0);
  EXPECT_EQ(float32_t(test), 0);
}

TEST_F(UtestFP16, Float64_t_success) {
  fp16_t test(0);
  EXPECT_EQ(float64_t(test), 0);
}

TEST_F(UtestFP16, Int8_t_success) {
  fp16_t test(0);
  EXPECT_EQ(int8_t(test), '\0');
  fp16_t test1(-1);
  EXPECT_EQ(int8_t(test1), -128);
  fp16_t test2(1024);
  EXPECT_EQ(int8_t(test2), '\0');
  fp16_t test3(31744);
  EXPECT_EQ(int8_t(test3), '\x7F');
  fp16_t test4((uint16_t)130944U);
  EXPECT_EQ(int8_t(test4), -128);
}

TEST_F(UtestFP16, Uint8_t_success) {
  fp16_t test(0);
  EXPECT_EQ(uint8_t(test), '\0');
  fp16_t test2(1024);
  EXPECT_EQ(uint8_t(test2), '\0');
  fp16_t test3(31744);
  EXPECT_EQ(uint8_t(test3), 255);
  fp16_t test4((uint16_t)130944U);
  EXPECT_EQ(uint8_t(test4), '\0');
}

TEST_F(UtestFP16, Int16_t_success) {
  fp16_t test(0);
  EXPECT_EQ(int16_t(test), 0);
  fp16_t test1(31744);
  EXPECT_EQ(int16_t(test1), 32767);
  fp16_t test2(64512);
  EXPECT_EQ(int16_t(test2), -32768);

  fp16_t test3(~kFp16ExpMask);
  EXPECT_EQ(int16_t(test3), 0);
}

TEST_F(UtestFP16, uint16_t_success) {
  fp16_t test(0);
  EXPECT_EQ(uint16_t(test), 0);
  fp16_t test1(31744);
  EXPECT_EQ(uint16_t(test1), 255);
  fp16_t test2(64512);
  EXPECT_EQ(uint16_t(test2), 0);
  fp16_t test3(~31744);
  EXPECT_EQ(uint16_t(test3), 0);
}

TEST_F(UtestFP16, Int32_t_success) {
  fp16_t test(0);
  EXPECT_EQ(int32_t(test), 0);

  fp16_t test1(kFp16ExpMask);
  EXPECT_EQ(int32_t(test1), 2147483647);

  fp16_t test2(~kFp16ExpMask);
  EXPECT_EQ(int32_t(test2), 0);
}

TEST_F(UtestFP16, Uint32_t_success) {
  fp16_t test(0);
  EXPECT_EQ(uint32_t(test), 0);
  fp16_t test1(64512);
  EXPECT_EQ(uint32_t(test1), 0);

  fp16_t test2(kFp16ExpMask);
  EXPECT_EQ(uint32_t(test2), 255);

  fp16_t test3(~kFp16ExpMask);
  EXPECT_EQ(uint32_t(test3), 0);
}

TEST_F(UtestFP16, Int64_t_success) {
  fp16_t test(0);
  EXPECT_EQ(int64_t(test), 0);
}

TEST_F(UtestFP16, uint64_t_success) {
  fp16_t test(0);
  EXPECT_EQ(uint64_t(test), 0);
}

TEST_F(UtestFP16, RightShift_success) {
  int16_t man = 0;
  fp16_t test;
  EXPECT_EQ(RightShift(man, 0), 0);
}

TEST_F(UtestFP16, GetManSum_success) {
  int16_t m_a = 0;
  int16_t m_b = 0;
  fp16_t test;
  EXPECT_EQ(GetManSum(0, m_a, 1, m_b), 0);
  EXPECT_EQ(GetManSum(1, m_a, 0, m_b), 0);
}
}  // namespace formats
}  // namespace ge

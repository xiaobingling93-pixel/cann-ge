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
#include "utils/node_expr_id.h"

namespace att {
namespace test {

class TestNodeExprId : public ::testing::Test {
 public:
  static void SetUpTestCase() {}
  static void TearDownTestCase() {}
  void SetUp() override {}
  void TearDown() override {}
};

// 测试用例 1：测试 GetExprVarPrefix 返回节点名_节点类型
TEST_F(TestNodeExprId, TestGetExprVarPrefix) {
  NodeExprId id;
  id.node_name = "mul_node";
  id.node_type = "Mul";
  id.input_shapes = {"[32,64]"};
  id.output_shapes = {"[32,64]"};

  std::string prefix = id.GetExprVarPrefix();
  EXPECT_EQ(prefix, "mul_node_Mul");
}

// 测试用例 2：测试 GetVarPrefix 格式（包含节点类型和形状信息）
TEST_F(TestNodeExprId, TestGetVarPrefixWithShapes) {
  NodeExprId id;
  id.node_name = "mul_node";
  id.node_type = "Mul";
  id.input_shapes = {"[32,64]"};
  id.output_shapes = {"[32,64]"};

  std::string prefix = id.GetVarPrefix();
  EXPECT_EQ(prefix, "mul_node_Mul_in[32,64]_out[32,64]");
}

// 测试用例 3：测试多个输入形状
TEST_F(TestNodeExprId, TestMultipleInputShapes) {
  NodeExprId id;
  id.node_name = "add_node";
  id.node_type = "Add";
  id.input_shapes = {"[32,64]", "[32,64]"};
  id.output_shapes = {"[32,64]"};

  std::string prefix = id.GetVarPrefix();
  EXPECT_EQ(prefix, "add_node_Add_in[32,64],[32,64]_out[32,64]");
}

// 测试用例 4：测试空形状情况
TEST_F(TestNodeExprId, TestEmptyShapes) {
  NodeExprId id;
  id.node_name = "scalar_node";
  id.node_type = "Scalar";
  // input_shapes 和 output_shapes 默认为空

  std::string prefix = id.GetVarPrefix();
  EXPECT_EQ(prefix, "scalar_node_Scalar_in[]_out[]");
}

// 测试用例 5：测试单输入单输出
TEST_F(TestNodeExprId, TestSingleShape) {
  NodeExprId id;
  id.node_name = "reduce_node";
  id.node_type = "Reduce";
  id.input_shapes = {"[1024]"};
  id.output_shapes = {"[1]"};

  std::string prefix = id.GetVarPrefix();
  EXPECT_EQ(prefix, "reduce_node_Reduce_in[1024]_out[1]");
}

// 测试用例 6：测试多维度形状
TEST_F(TestNodeExprId, TestMultiDimShapes) {
  NodeExprId id;
  id.node_name = "conv2d_node";
  id.node_type = "Conv2D";
  id.input_shapes = {"[1,64,56,56]", "[64,64,3,3]"};
  id.output_shapes = {"[1,64,56,56]"};

  std::string prefix = id.GetVarPrefix();
  EXPECT_EQ(prefix, "conv2d_node_Conv2D_in[1,64,56,56],[64,64,3,3]_out[1,64,56,56]");
}

// 测试用例 7：测试变量名使用节点名_节点类型格式
TEST_F(TestNodeExprId, TestUseNodeNameWithType) {
  NodeExprId id;
  id.node_name = "my_custom_mul";
  id.node_type = "Mul";
  id.input_shapes = {"[32,64]"};
  id.output_shapes = {"[32,64]"};

  std::string expr_prefix = id.GetExprVarPrefix();
  std::string var_prefix = id.GetVarPrefix();

  // 验证表达式变量名是 节点名_节点类型
  EXPECT_EQ(expr_prefix, "my_custom_mul_Mul");
  // 验证完整前缀包含节点名、节点类型和形状
  EXPECT_EQ(var_prefix, "my_custom_mul_Mul_in[32,64]_out[32,64]");
}

// 测试用例 8：测试不同节点名产生不同前缀
TEST_F(TestNodeExprId, TestDifferentNodeNames) {
  NodeExprId id1;
  id1.node_name = "load1";
  id1.node_type = "Load";
  id1.input_shapes = {"[32,64]"};
  id1.output_shapes = {"[32,64]"};

  NodeExprId id2;
  id2.node_name = "load2";
  id2.node_type = "Load";
  id2.input_shapes = {"[32,64]"};
  id2.output_shapes = {"[32,64]"};

  std::string prefix1 = id1.GetVarPrefix();
  std::string prefix2 = id2.GetVarPrefix();

  EXPECT_EQ(prefix1, "load1_Load_in[32,64]_out[32,64]");
  EXPECT_EQ(prefix2, "load2_Load_in[32,64]_out[32,64]");

  // 验证节点名不同
  EXPECT_NE(prefix1, prefix2);
  EXPECT_TRUE(prefix1.find("load1_Load") != std::string::npos);
  EXPECT_TRUE(prefix2.find("load2_Load") != std::string::npos);
}

// 测试用例 9：测试相同节点名但不同节点类型
TEST_F(TestNodeExprId, TestSameNameDifferentType) {
  NodeExprId id1;
  id1.node_name = "op1";
  id1.node_type = "Load";
  id1.input_shapes = {"[32,64]"};
  id1.output_shapes = {"[32,64]"};

  NodeExprId id2;
  id2.node_name = "op1";
  id2.node_type = "Store";
  id2.input_shapes = {"[32,64]"};
  id2.output_shapes = {"[32,64]"};

  std::string prefix1 = id1.GetVarPrefix();
  std::string prefix2 = id2.GetVarPrefix();

  EXPECT_EQ(prefix1, "op1_Load_in[32,64]_out[32,64]");
  EXPECT_EQ(prefix2, "op1_Store_in[32,64]_out[32,64]");

  // 验证节点类型不同导致前缀不同
  EXPECT_NE(prefix1, prefix2);
}

} // namespace test
} // namespace att

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
#include <string>
#include <vector>
#include "es_graph_builder.h"
#include "es_tensor_holder.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/file_utils.h"
#include "c_generator.h"
#include "cpp_generator.h"
#include "utils.h"
#include "graph/op_desc.h"
#include "es_codegen_default_value.h"
#include "history/history_registry_utils.h"
#include "ge_running_env/path_utils.h"
#include "gen_esb_options.h"

namespace ge {
namespace es {
void GenEsImpl(const GenEsbOptions &options);
}  // namespace es
}  // namespace ge
namespace {
auto Normalize = [](const std::string &code) {
  std::string str = code;
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  return str;
};

template <typename F>
void ExpectInvalidArgumentErrorContains(F &&fn, const std::string &expected_msg) {
  try {
    fn();
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find(expected_msg), std::string::npos);
  } catch (...) {
    FAIL() << "Expected std::invalid_argument";
  }
}

}
class GenImplLLT : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建测试输出目录
    test_output_dir_ = "./test_gen_output/";
    ge::es::GenEsbOptions opts;
    opts.mode = ge::es::kEsCodeGenDefaultMode;
    opts.output_dir = test_output_dir_;
    opts.module_name = ge::es::kEsCodeGenDefaultModelName;
    opts.h_guard_prefix = ge::es::kEsCodeGenDefaultPrefixGuard;
    opts.exclude_ops = "phony_all";
    ge::es::GenEsImpl(opts);
  }

  void TearDown() override {
    const std::string command = "rm -rf " + test_output_dir_;
    (void) std::system(command.c_str());
  }

  std::string test_output_dir_;
};

class GenImplLLTExtractHistory : public ::testing::Test {
 protected:
  void SetUp() override {
    opts.mode = ge::es::kEsExtractHistoryMode;
    opts.output_dir = "./test_history_gen_output_" + std::to_string(getpid()) + "/";
  }

  void TearDown() override {
    const std::string command = "rm -rf " + opts.output_dir;
    (void) std::system(command.c_str());
  }

  ge::es::GenEsbOptions opts;
};

// 测试生成的文件结构
TEST_F(GenImplLLT, GeneratedFileStructure) {
  // 模拟生成的文件
  std::vector<std::string> expected_files = {"es_all_ops_c.h",        "es_all_ops.h",         "es_phony_1i_1o.h",
                                             "es_phony_1i_1o.cpp",     "es_phony_1i_1o_c.h",    "es_phony_1i1dyi_1o.h",
                                             "es_phony_1i1dyi_1o.cpp", "es_phony_1i1dyi_1o_c.h"};
  // 验证文件是否生成
  for (const auto &file : expected_files) {
    std::string file_path = test_output_dir_ + file;
    EXPECT_FALSE(ge::RealPath(file_path.c_str()).empty()) << "File should exist: " << file_path;
  }
}

// 测试文件名生成
TEST_F(GenImplLLT, FileNameGeneration) {
  std::string op_type = "phony_1i_1o";

  // 测试C头文件名生成
  std::string c_header_name = ge::es::CGenerator::PerOpHeaderFileName(op_type);
  EXPECT_EQ(c_header_name, "es_phony_1i_1o_c.h");

  std::string c_source_name = ge::es::CGenerator::PerOpSourceFileName(op_type);
  EXPECT_EQ(c_source_name, "es_phony_1i_1o.cpp");

  // 测试C++头文件名生成
  std::string cpp_header_name = ge::es::CppGenerator::PerOpHeaderFileName(op_type);
  EXPECT_EQ(cpp_header_name, "es_phony_1i_1o.h");
}

// 测试工具函数
TEST_F(GenImplLLT, UtilityFunctions) {
  auto builder = std::make_unique<ge::es::EsGraphBuilder>("tensor_holder_test");
  auto tensor = builder->CreateScalar(int64_t(123));
  auto node = ge::NodeAdapter::GNode2Node(*tensor.GetProducer());
  auto opdesc = node->GetOpDesc();
  // 测试关键字检测
  EXPECT_TRUE(ge::es::IsKeyword("class"));
  EXPECT_TRUE(ge::es::IsKeyword("int"));
  EXPECT_TRUE(ge::es::IsKeyword("return"));
  EXPECT_FALSE(ge::es::IsKeyword("phony_1i_1o"));
  EXPECT_FALSE(ge::es::IsKeyword(""));

  // 测试名称转换
  EXPECT_EQ(ge::es::InName("class"), "in_class");
  EXPECT_EQ(ge::es::InName("normal_name"), "normal_name");
  EXPECT_EQ(ge::es::OutName("int", opdesc), "out_int");
  EXPECT_EQ(ge::es::OutName("output", opdesc), "output");
  EXPECT_EQ(ge::es::AttrName("return", opdesc), "attr_return");
  EXPECT_EQ(ge::es::AttrName("index", opdesc), "index");
  EXPECT_TRUE(ge::es::IsOpSkip("Data"));
  EXPECT_TRUE(ge::es::IsOpSkip("NETOUTPUT"));
  EXPECT_FALSE(ge::es::IsOpSkip("Relu"));
  std::stringstream ss;
  ge::es::WriteOut(nullptr, ss);
}

// 测试聚合头文件的生成
TEST_F(GenImplLLT, AggregateHeaderGeneration) {
  // 模拟生成的all_ops_c.h文件内容
  std::string expected_all_ops_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_all_OPS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_all_OPS_H_
#ifdef __cplusplus
extern "C" {
#endif

/** Unsupported ops and reasons:
 * Unexpected data type: -1(1): [phony_invalid_attr_value, ]
 * Following Operators already provided creation functions(1): [Variable, ]
 * Does not support V1 control Operators(8): [Enter, Exit, LoopCond, Merge, NextIteration, StreamMerge, StreamSwitch, Switch, ]
 */
#include "es_Add_c.h"
#include "es_AscBackend_c.h"
#include "es_AscBackendNoKernelOp_c.h"
#include "es_AscGraph_c.h"
#include "es_Const_c.h"
#include "es_FusedAscBackend_c.h"
#include "es_While_c.h"
#include "es_phony_1i1dyi_1dyo_c.h"
#include "es_phony_1i1dyi_1o_c.h"
#include "es_phony_1i1dyi_2o2dyo1o_c.h"
#include "es_phony_1i1dyi_3dyo_c.h"
#include "es_phony_1i1opi_1o_c.h"
#include "es_phony_1i_1dyo_c.h"
#include "es_phony_1i_1dyo1o_c.h"
#include "es_phony_1i_1o_c.h"
#include "es_phony_1i_1o1dyo_c.h"
#include "es_phony_1i_2dyo_c.h"
#include "es_phony_1i_2o_c.h"
#include "es_phony_1i_2o1dyo_c.h"
#include "es_phony_1opi1i_1o_c.h"
#include "es_phony_1opi_1o_c.h"
#include "es_phony_2opi1i_1o_c.h"
#include "es_phony_3opi_1o_c.h"
#include "es_phony_Case_c.h"
#include "es_phony_If_c.h"
#include "es_phony_PartitionedCall_c.h"
#include "es_phony_dup_name_c.h"
#include "es_phony_mix_subgraphs_c.h"
#include "es_phony_multi_attr_c.h"
#include "es_phony_opt_attrs_c.h"
#include "es_phony_req_attrs_c.h"
#include "es_phony_same_name_c.h"

#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证聚合头文件内容 - 直接比较内容是否一致
  std::ifstream read_all_ops(test_output_dir_ + "es_all_ops_c.h");
  std::stringstream all_ops_content;
  all_ops_content << read_all_ops.rdbuf();
  read_all_ops.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(all_ops_content.str()), Normalize(expected_all_ops_content))
      << "Generated aggregate header content does not match expected content";
}

// 测试PY文件生成
TEST_F(GenImplLLT, AggregatePyGeneration) {
  std::string expected_py_content = R"(
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ===================================================================================================================
""" This file is GENERATED by bin/gen_esb, do not edit it manually"""

""" Unsupported ops and reasons:
 # Unexpected data type: -1(1): [phony_invalid_attr_value, ]
 # Following Operators already provided creation functions(1): [Variable, ]
 # Does not support V1 control Operators(8): [Enter, Exit, LoopCond, Merge, NextIteration, StreamMerge, StreamSwitch, Switch, ]
"""
from .es_Add import *
from .es_AscBackendimport *
from .es_AscBackendNoKernelOpimport *
from .es_AscGraphimport *
from .es_Const import *
from .es_FusedAscBackendimport *
from .es_While import *
from .es_phony_1i1dyi_1dyo import *
from .es_phony_1i1dyi_1o import *
from .es_phony_1i1dyi_2o2dyo1o import *
from .es_phony_1i1dyi_3dyo import *
from .es_phony_1i1opi_1o import *
from .es_phony_1i_1dyo import *
from .es_phony_1i_1dyo1o import *
from .es_phony_1i_1o import *
from .es_phony_1i_1o1dyo import *
from .es_phony_1i_2dyo import *
from .es_phony_1i_2o import *
from .es_phony_1i_2o1dyo import *
from .es_phony_1opi1i_1o import *
from .es_phony_1opi_1o import *
from .es_phony_2opi1i_1o import *
from .es_phony_3opi_1o import *
from .es_phony_Case import *
from .es_phony_If import *
from .es_phony_PartitionedCall import *
from .es_phony_dup_name import *
from .es_phony_mix_subgraphs import *
from .es_phony_multi_attr import *
from .es_phony_opt_attrs import *
from .es_phony_req_attrs import *
from .es_phony_same_name import *

)";

  // 验证聚合头文件内容 - 直接比较内容是否一致
  std::ifstream read_all_ops(test_output_dir_ + "es_all_ops.py");
  std::stringstream all_ops_content;
  all_ops_content << read_all_ops.rdbuf();
  read_all_ops.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(all_ops_content.str()), Normalize(expected_py_content))
      << "Generated aggregate py content does not match expected content";
}

// 测试Add算子
TEST_F(GenImplLLT, Add_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，Add应该有以下特征：
  // REG_OP(Add)
  //     .INPUT(x1, TensorType::All())
  //     .INPUT(x2, TensorType::All())
  //     .OUTPUT(y, TensorType::All())
  //     .OP_END_FACTORY_REG(Add);

  // 模拟生成的es_Add_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_Add_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_Add_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *EsAdd(EsCTensorHolder *x1, EsCTensorHolder *x2);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_Add_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, Add_OpCppGeneration) {
  // 模拟生成的Add.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_Add_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *EsAdd(EsCTensorHolder *x1, EsCTensorHolder *x2) {
  ES_ASSERT_NOTNULL(x1);
  ES_ASSERT_NOTNULL(x2);
  auto *owner_builder = ge::es::ResolveBuilder(x1, x2);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("Add")
      .Name( builder.GenerateNodeName("Add").GetString())
      .IrDefInputs({
          {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
      })
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x1->GetProducer(), x1->GetOutIndex(), node, 0));
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x2->GetProducer(), x2->GetOutIndex(), node, 1));
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_Add.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, Add_OpCppHeaderGeneration) {
  // 模拟生成的Add头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_Add_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_Add_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_Add_c.h"
namespace ge {
namespace es {

/**
 * @note at least one of the following input arguments should be EsTensorHolder object:
 *   x1
 *   x2
 */
inline EsTensorHolder Add(const EsTensorLike &x1, const EsTensorLike &x2) {
  auto *owner_graph_builder = ResolveBuilder(x1, x2);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto out = EsAdd(x1.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), x2.ToTensorHolder(owner_graph_builder).GetCTensorHolder());
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_Add.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

TEST_F(GenImplLLT, phony_1i_1o_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i_1o应该有以下特征：
  // - 1个输入 (x)
  // - 1个输出 (y)
  // - 1个属性 (index, Int类型，默认值0)

  // 模拟生成的phony_1i_1o_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_1o_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_1o_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_1i_1o(EsCTensorHolder *x, int64_t index);
#ifdef __cplusplus
}
#endif
#endif
)";
  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i_1o_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i_1o_OpCppGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i_1o应该有以下特征：
  // - 1个输入 (x)
  // - 1个输出 (y)
  // - 1个属性 (index, Int类型，默认值0)
  // 模拟生成的phony_1i_1o.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_1i_1o_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_1i_1o(EsCTensorHolder *x, int64_t index) {
  ES_ASSERT_NOTNULL(x);
  auto *owner_builder = ge::es::ResolveBuilder(x);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_1i_1o")
      .Name( builder.GenerateNodeName("phony_1i_1o").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {
              "index",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Int",
              ge::es::CreateFromIfNotEqual(index, static_cast<int64_t>(0))
          },
      })
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i_1o.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

// 测试C++包装器的生成
TEST_F(GenImplLLT, CppWrapperGeneration) {
  // 模拟生成的phony_1i_1o.h文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_1o_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_1o_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_phony_1i_1o_c.h"
namespace ge {
namespace es {

inline EsTensorHolder phony_1i_1o(const EsTensorHolder &x, int64_t index=0) {
  auto out = Esphony_1i_1o(x.GetCTensorHolder(), index);
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i_1o.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试phony_1i1dyi_1o操作符的生成
TEST_F(GenImplLLT, phony_1i1dyi_1o_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i1dyi_1o应该有以下特征：
  // - 1个输入 (x)
  // - 1个动态输入 (dx)
  // - 1个输出 (y)
  // - 1个属性 (index, ListInt类型，默认值{10, 10, 10})

  // 模拟生成的es_phony_1i1dyi_1o_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_1o_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_1o_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_1i1dyi_1o(EsCTensorHolder *x, EsCTensorHolder **dx, int64_t dx_num, const int64_t *index, int64_t index_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i1dyi_1o_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i1dyi_1o_OpCppGeneration) {
  // 模拟生成的phony_1i1dyi_1o.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_1i1dyi_1o_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_1i1dyi_1o(EsCTensorHolder *x, EsCTensorHolder **dx, int64_t dx_num, const int64_t *index, int64_t index_num) {
  ES_ASSERT_NOTNULL(x);
  ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(dx_num));
  auto *owner_builder = ge::es::ResolveBuilder(x, std::vector<EsCTensorHolder *>(dx, dx + dx_num));
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_1i1dyi_1o")
      .Name( builder.GenerateNodeName("phony_1i1dyi_1o").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"dx", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {
              "index",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListInt",
              ge::es::CreateFromIfNotEqual(std::vector<int64_t>(index, index + index_num), std::vector<int64_t>{10, 10, 10})
          },
      })
      .InstanceDynamicInputNum("dx", static_cast<int32_t>(dx_num))
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  if ((dx != nullptr) && (dx_num > 0)) {
    for (int64_t i = 0; i < dx_num; ++i) {
      auto one_dx = dx[i];
      ES_ASSERT_NOTNULL(one_dx);
      ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, one_dx->GetProducer(), one_dx->GetOutIndex(), node, 1 + i));
    }
  }
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i1dyi_1o.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i1dyi_1o_OpCppHeaderGeneration) {
  // 模拟生成的phony_1i1dyi_1o头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_1o_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_1o_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_1i1dyi_1o_c.h"
namespace ge {
namespace es {

inline EsTensorHolder phony_1i1dyi_1o(const EsTensorLike &x, const std::vector<EsTensorHolder> &dx, const std::vector<int64_t> &index={10, 10, 10}) {
  auto *owner_graph_builder = ResolveBuilder(x, dx);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto esb_dx = TensorsToEsCTensorHolders(dx);
  auto out = Esphony_1i1dyi_1o(x.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), esb_dx.data(), static_cast<int64_t>(esb_dx.size()), index.data(), static_cast<int64_t>(index.size()));
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i1dyi_1o.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试可选必选调整顺序
TEST_F(GenImplLLT, phony_1i1opi_1o_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i1opi_1o应该有以下特征：
  // - 1个输入 (x)
  // - 1个可选输入 (dx)
  // - 1个输出 (y)
  // - 1个属性 (dt, Type类型，默认值DT_FLOAT)
  // - 1个必需属性 (flag, Bool类型)

  // 模拟生成的phony_1i1opi_1o_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1opi_1o_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1opi_1o_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_1i1opi_1o(EsCTensorHolder *x, EsCTensorHolder *dx, bool flag, C_DataType dt);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i1opi_1o_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i1opi_1o_OpCppGeneration) {
  // 模拟生成的phony_1i1opi_1o.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_1i1opi_1o_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_1i1opi_1o(EsCTensorHolder *x, EsCTensorHolder *dx, bool flag, C_DataType dt) {
  ES_ASSERT_NOTNULL(x);
  auto *owner_builder = ge::es::ResolveBuilder(x, dx);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_1i1opi_1o")
      .Name( builder.GenerateNodeName("phony_1i1opi_1o").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"dx", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {
              "dt",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Type",
              ge::es::CreateFromIfNotEqual(static_cast<ge::DataType>(dt), ge::DT_FLOAT)
          },
          {
              "flag",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "Bool",
              ge::es::CreateFrom(static_cast<bool>(flag))
          },
      })
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  if (dx != nullptr) {
    ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, dx->GetProducer(), dx->GetOutIndex(), node, 1));
  }
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i1opi_1o.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i1opi_1o_OpCppHeaderGeneration) {
  // 模拟生成的phony_1i1opi_1o头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1opi_1o_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1opi_1o_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_1i1opi_1o_c.h"
namespace ge {
namespace es {

/**
 * @note at least one of the following input arguments should be EsTensorHolder object:
 *   x
 *   dx
 */
inline EsTensorHolder phony_1i1opi_1o(const EsTensorLike &x, const EsTensorLike &dx, bool flag, ge::DataType dt=ge::DT_FLOAT) {
  auto *owner_graph_builder = ResolveBuilder(x, dx);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto out = Esphony_1i1opi_1o(x.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), dx.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), flag, static_cast<C_DataType>(dt));
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i1opi_1o.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试1i2o情况
TEST_F(GenImplLLT, phony_1i_2o_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i_2o应该有以下特征：
  // REG_OP(phony_1i_2o)
  //     .INPUT(x, TensorType::All())
  //     .OUTPUT(y1, TensorType::All())
  //     .OUTPUT(y2, TensorType::All())
  //     .OP_END_FACTORY_REG(phony_1i_2o);

  // 模拟生成的es_phony_1i_2o_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_2o_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_2o_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder *y1;
  EsCTensorHolder *y2;
} Esphony_1i_2oOutput;
Esphony_1i_2oOutput Esphony_1i_2o(EsCTensorHolder *x);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i_2o_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i_2o_OpCppGeneration) {
  // 模拟生成的phony_1i_2o.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_1i_2o_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
Esphony_1i_2oOutput Esphony_1i_2o(EsCTensorHolder *x) {
  ES_ASSERT_NOTNULL(x);
  auto *owner_builder = ge::es::ResolveBuilder(x);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_1i_2o")
      .Name( builder.GenerateNodeName("phony_1i_2o").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"y1", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
          {"y2", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
      })
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  return Esphony_1i_2oOutput{
      builder.GetTensorHolderFromNode(node, 0),
      builder.GetTensorHolderFromNode(node, 1),
};
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i_2o.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i_2o_OpCppHeaderGeneration) {
  // 模拟生成的phony_1i_2o头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_2o_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_2o_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_phony_1i_2o_c.h"
namespace ge {
namespace es {

struct phony_1i_2oOutput {
  EsTensorHolder y1;
  EsTensorHolder y2;
};
inline phony_1i_2oOutput phony_1i_2o(const EsTensorHolder &x) {
  auto out = Esphony_1i_2o(x.GetCTensorHolder());
  return {out.y1, out.y2};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i_2o.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试动态输出--单动态输出算子
TEST_F(GenImplLLT, phony_1i1dyi_1dyo_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i1dyi_1dyo应该有以下特征：
  // REG_OP(phony_1i1dyi_1dyo)
  //     .INPUT(x, TensorType::All())
  //     .DYNAMIC_INPUT(dx, TensorType::All())
  //     .DYNAMIC_OUTPUT(dy, TensorType::All())
  //     .OP_END_FACTORY_REG(phony_1i1dyi_1dyo);

  // 模拟生成的es_phony_1i1dyi_1dyo_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_1dyo_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_1dyo_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **dy;
  int64_t dy_num;
} Esphony_1i1dyi_1dyoOutput;
/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy_num: dynamic output number of dy
 */
Esphony_1i1dyi_1dyoOutput Esphony_1i1dyi_1dyo(EsCTensorHolder *x, EsCTensorHolder **dx, int64_t dx_num, int64_t dy_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i1dyi_1dyo_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i1dyi_1dyo_OpCppHeaderGeneration) {
  // 模拟生成的phony_1i1dyi_1dyo头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_1dyo_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_1dyo_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_1i1dyi_1dyo_c.h"
namespace ge {
namespace es {

/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy_num: dynamic output number of dy
 */
inline std::vector<EsTensorHolder> phony_1i1dyi_1dyo(const EsTensorLike &x, const std::vector<EsTensorHolder> &dx, int64_t dy_num) {
  auto *owner_graph_builder = ResolveBuilder(x, dx);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto esb_dx = TensorsToEsCTensorHolders(dx);
  auto out = Esphony_1i1dyi_1dyo(x.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), esb_dx.data(), static_cast<int64_t>(esb_dx.size()), dy_num);
  std::vector<EsTensorHolder> dy_dynamic_outs;
  dy_dynamic_outs.reserve(out.dy_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy_num; ++dyn_idx) {
    dy_dynamic_outs.emplace_back(out.dy[dyn_idx]);
  }
  return {dy_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i1dyi_1dyo.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试动态输出--单动态输出+单输出算子
TEST_F(GenImplLLT, phony_1i_1o1dyo_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i_1o1dyo应该有以下特征：
  // REG_OP(phony_1i_1o1dyo)
  //   .INPUT(x, TensorType::All())
  //   .OUTPUT(y, TensorType::All())
  //   .DYNAMIC_OUTPUT(dy, TensorType::All())
  //   .OP_END_FACTORY_REG(phony_1i_1o1dyo);

  // 模拟生成的es_phony_1i_1o1dyo_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_1o1dyo_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_1o1dyo_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder *y;
  EsCTensorHolder **dy;
  int64_t dy_num;
} Esphony_1i_1o1dyoOutput;
/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy_num: dynamic output number of dy
 */
Esphony_1i_1o1dyoOutput Esphony_1i_1o1dyo(EsCTensorHolder *x, int64_t dy_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i_1o1dyo_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i_1o1dyo_OpCppGeneration) {
  // 模拟生成的phony_1i_1o1dyo.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_1i_1o1dyo_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
Esphony_1i_1o1dyoOutput Esphony_1i_1o1dyo(EsCTensorHolder *x, int64_t dy_num) {
  ES_ASSERT_NOTNULL(x);
  auto *owner_builder = ge::es::ResolveBuilder(x);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_1i_1o1dyo")
      .Name( builder.GenerateNodeName("phony_1i_1o1dyo").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
          {"dy", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
      })
      .IrDefAttrs({
      })
      .InstanceDynamicOutputNum("dy", static_cast<int32_t>(dy_num))
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  auto dy_holders = builder.CreateDynamicTensorHolderFromNode(node, 1, dy_num);
  return Esphony_1i_1o1dyoOutput{
      builder.GetTensorHolderFromNode(node, 0),
      dy_holders->data(),
      static_cast<int64_t>(dy_holders->size()),
  };
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i_1o1dyo.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i_1o1dyo_OpCppHeaderGeneration) {
  // 模拟生成的phony_1i_1o1dyo头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_1o1dyo_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_1o1dyo_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_phony_1i_1o1dyo_c.h"
namespace ge {
namespace es {

struct phony_1i_1o1dyoOutput {
  EsTensorHolder y;
  std::vector<EsTensorHolder> dy;
};
/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy_num: dynamic output number of dy
 */
inline phony_1i_1o1dyoOutput phony_1i_1o1dyo(const EsTensorHolder &x, int64_t dy_num) {
  auto out = Esphony_1i_1o1dyo(x.GetCTensorHolder(), dy_num);
  std::vector<EsTensorHolder> dy_dynamic_outs;
  dy_dynamic_outs.reserve(out.dy_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy_num; ++dyn_idx) {
    dy_dynamic_outs.emplace_back(out.dy[dyn_idx]);
  }
  return {out.y, dy_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i_1o1dyo.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试动态输出--单动态输出+多输出算子
TEST_F(GenImplLLT, phony_1i_2o1dyo_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i_2o1dyo应该有以下特征：
  // REG_OP(phony_1i_2o1dyo)
  //     .INPUT(x, TensorType::All())
  //     .OUTPUT(y1, TensorType::All())
  //     .OUTPUT(y2, TensorType::All())
  //     .DYNAMIC_OUTPUT(dy, TensorType::All())
  //     .OP_END_FACTORY_REG(phony_1i_2o1dyo);

  // 模拟生成的es_phony_1i_2o1dyo_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_2o1dyo_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_2o1dyo_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder *y1;
  EsCTensorHolder *y2;
  EsCTensorHolder **dy;
  int64_t dy_num;
} Esphony_1i_2o1dyoOutput;
/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy_num: dynamic output number of dy
 */
Esphony_1i_2o1dyoOutput Esphony_1i_2o1dyo(EsCTensorHolder *x, int64_t dy_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i_2o1dyo_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i_2o1dyo_OpCppGeneration) {
  // 模拟生成的phony_1i_2o1dyo.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_1i_2o1dyo_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
Esphony_1i_2o1dyoOutput Esphony_1i_2o1dyo(EsCTensorHolder *x, int64_t dy_num) {
  ES_ASSERT_NOTNULL(x);
  auto *owner_builder = ge::es::ResolveBuilder(x);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_1i_2o1dyo")
      .Name( builder.GenerateNodeName("phony_1i_2o1dyo").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"y1", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
          {"y2", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
          {"dy", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
      })
      .IrDefAttrs({
      })
      .InstanceDynamicOutputNum("dy", static_cast<int32_t>(dy_num))
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  auto dy_holders = builder.CreateDynamicTensorHolderFromNode(node, 2, dy_num);
  return Esphony_1i_2o1dyoOutput{
      builder.GetTensorHolderFromNode(node, 0),
      builder.GetTensorHolderFromNode(node, 1),
      dy_holders->data(),
      static_cast<int64_t>(dy_holders->size()),
  };
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i_2o1dyo.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i_2o1dyo_OpCppHeaderGeneration) {
  // 模拟生成的phony_1i_2o1dyo头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_2o1dyo_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i_2o1dyo_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_phony_1i_2o1dyo_c.h"
namespace ge {
namespace es {

struct phony_1i_2o1dyoOutput {
  EsTensorHolder y1;
  EsTensorHolder y2;
  std::vector<EsTensorHolder> dy;
};
/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy_num: dynamic output number of dy
 */
inline phony_1i_2o1dyoOutput phony_1i_2o1dyo(const EsTensorHolder &x, int64_t dy_num) {
  auto out = Esphony_1i_2o1dyo(x.GetCTensorHolder(), dy_num);
  std::vector<EsTensorHolder> dy_dynamic_outs;
  dy_dynamic_outs.reserve(out.dy_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy_num; ++dyn_idx) {
    dy_dynamic_outs.emplace_back(out.dy[dyn_idx]);
  }
  return {out.y1, out.y2, dy_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i_2o1dyo.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试动态输出--多动态输出算子
TEST_F(GenImplLLT, phony_1i1dyi_3dyo_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i1dyi_3dyo应该有以下特征：
  // REG_OP(phony_1i1dyi_3dyo)
  //     .INPUT(x, TensorType::All())
  //     .DYNAMIC_INPUT(dx, TensorType::All())
  //     .DYNAMIC_OUTPUT(dy1, TensorType::All())
  //     .DYNAMIC_OUTPUT(dy2, TensorType::All())
  //     .DYNAMIC_OUTPUT(dy3, TensorType::All())
  //     .OP_END_FACTORY_REG(phony_1i1dyi_3dyo);

  // 模拟生成的es_phony_1i1dyi_3dyo_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_3dyo_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_3dyo_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **dy1;
  int64_t dy1_num;
  EsCTensorHolder **dy2;
  int64_t dy2_num;
  EsCTensorHolder **dy3;
  int64_t dy3_num;
} Esphony_1i1dyi_3dyoOutput;
/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy1_num: dynamic output number of dy1
 *   dy2_num: dynamic output number of dy2
 *   dy3_num: dynamic output number of dy3
 */
Esphony_1i1dyi_3dyoOutput Esphony_1i1dyi_3dyo(EsCTensorHolder *x, EsCTensorHolder **dx, int64_t dx_num, int64_t dy1_num, int64_t dy2_num, int64_t dy3_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i1dyi_3dyo_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i1dyi_3dyo_OpCppHeaderGeneration) {
  // 模拟生成的phony_1i1dyi_3dyo头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_3dyo_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_3dyo_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_1i1dyi_3dyo_c.h"
namespace ge {
namespace es {

struct phony_1i1dyi_3dyoOutput {
  std::vector<EsTensorHolder> dy1;
  std::vector<EsTensorHolder> dy2;
  std::vector<EsTensorHolder> dy3;
};
/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy1_num: dynamic output number of dy1
 *   dy2_num: dynamic output number of dy2
 *   dy3_num: dynamic output number of dy3
 */
inline phony_1i1dyi_3dyoOutput phony_1i1dyi_3dyo(const EsTensorLike &x, const std::vector<EsTensorHolder> &dx, int64_t dy1_num, int64_t dy2_num, int64_t dy3_num) {
  auto *owner_graph_builder = ResolveBuilder(x, dx);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto esb_dx = TensorsToEsCTensorHolders(dx);
  auto out = Esphony_1i1dyi_3dyo(x.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), esb_dx.data(), static_cast<int64_t>(esb_dx.size()), dy1_num, dy2_num, dy3_num);
  std::vector<EsTensorHolder> dy1_dynamic_outs;
  dy1_dynamic_outs.reserve(out.dy1_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy1_num; ++dyn_idx) {
    dy1_dynamic_outs.emplace_back(out.dy1[dyn_idx]);
  }
  std::vector<EsTensorHolder> dy2_dynamic_outs;
  dy2_dynamic_outs.reserve(out.dy2_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy2_num; ++dyn_idx) {
    dy2_dynamic_outs.emplace_back(out.dy2[dyn_idx]);
  }
  std::vector<EsTensorHolder> dy3_dynamic_outs;
  dy3_dynamic_outs.reserve(out.dy3_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy3_num; ++dyn_idx) {
    dy3_dynamic_outs.emplace_back(out.dy3[dyn_idx]);
  }
  return {dy1_dynamic_outs, dy2_dynamic_outs, dy3_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i1dyi_3dyo.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试动态输出--混合输出算子
TEST_F(GenImplLLT, phony_1i1dyi_2o2dyo1o_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1i1dyi_2o2dyo1o应该有以下特征：
  // REG_OP(phony_1i1dyi_2o2dyo1o)
  //     .INPUT(x, TensorType::All())
  //     .DYNAMIC_INPUT(dx, TensorType::All())
  //     .OUTPUT(y1, TensorType::All())
  //     .OUTPUT(y2, TensorType::All())
  //     .DYNAMIC_OUTPUT(dy1, TensorType::All())
  //     .DYNAMIC_OUTPUT(dy2, TensorType::All())
  //     .OUTPUT(y3, TensorType::All())
  //     .DYNAMIC_OUTPUT(dy3, TensorType::All())
  //     .ATTR(index, ListInt, {10, 10, 10})
  //     .ATTR(value, Tensor, Tensor())
  //     .OP_END_FACTORY_REG(phony_1i1dyi_2o2dyo1o);

  // 模拟生成的es_phony_1i1dyi_2o2dyo1o_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_2o2dyo1o_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_2o2dyo1o_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder *y1;
  EsCTensorHolder *y2;
  EsCTensorHolder **dy1;
  int64_t dy1_num;
  EsCTensorHolder **dy2;
  int64_t dy2_num;
  EsCTensorHolder *y3;
  EsCTensorHolder **dy3;
  int64_t dy3_num;
} Esphony_1i1dyi_2o2dyo1oOutput;
/**
 * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:
 *   value
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy1_num: dynamic output number of dy1
 *   dy2_num: dynamic output number of dy2
 *   dy3_num: dynamic output number of dy3
 */
Esphony_1i1dyi_2o2dyo1oOutput Esphony_1i1dyi_2o2dyo1o(EsCTensorHolder *x, EsCTensorHolder **dx, int64_t dx_num, int64_t dy1_num, int64_t dy2_num, int64_t dy3_num, const int64_t *index, int64_t index_num, EsCTensor *value);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i1dyi_2o2dyo1o_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i1dyi_2o2dyo1o_OpCppGeneration) {
  // 模拟生成的phony_1i1dyi_2o2dyo1o.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_1i1dyi_2o2dyo1o_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
Esphony_1i1dyi_2o2dyo1oOutput Esphony_1i1dyi_2o2dyo1o(EsCTensorHolder *x, EsCTensorHolder **dx, int64_t dx_num, int64_t dy1_num, int64_t dy2_num, int64_t dy3_num, const int64_t *index, int64_t index_num, EsCTensor *value) {
  ES_ASSERT_NOTNULL(x);
  ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(dx_num));
  auto *owner_builder = ge::es::ResolveBuilder(x, std::vector<EsCTensorHolder *>(dx, dx + dx_num));
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto value_stored = builder.AddResource(std::unique_ptr<ge::Tensor>(static_cast<ge::Tensor *>(static_cast<void *>(value))));
  ES_ASSERT_NOTNULL(value_stored);
  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_1i1dyi_2o2dyo1o")
      .Name( builder.GenerateNodeName("phony_1i1dyi_2o2dyo1o").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"dx", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
      })
      .IrDefOutputs({
          {"y1", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
          {"y2", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
          {"dy1", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
          {"dy2", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
          {"y3", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
          {"dy3", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
      })
      .IrDefAttrs({
          {
              "index",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListInt",
              ge::es::CreateFromIfNotEqual(std::vector<int64_t>(index, index + index_num), std::vector<int64_t>{10, 10, 10})
          },
          {
              "value",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Tensor",
              ge::es::CreateFrom(*value_stored)
          },
      })
      .InstanceDynamicInputNum("dx", static_cast<int32_t>(dx_num))
      .InstanceDynamicOutputNum("dy1", static_cast<int32_t>(dy1_num))
      .InstanceDynamicOutputNum("dy2", static_cast<int32_t>(dy2_num))
      .InstanceDynamicOutputNum("dy3", static_cast<int32_t>(dy3_num))
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  if ((dx != nullptr) && (dx_num > 0)) {
    for (int64_t i = 0; i < dx_num; ++i) {
      auto one_dx = dx[i];
      ES_ASSERT_NOTNULL(one_dx);
      ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, one_dx->GetProducer(), one_dx->GetOutIndex(), node, 1 + i));
    }
  }
  auto dy1_holders = builder.CreateDynamicTensorHolderFromNode(node, 2, dy1_num);
  auto dy2_holders = builder.CreateDynamicTensorHolderFromNode(node, 2 + dy1_num, dy2_num);
  auto dy3_holders = builder.CreateDynamicTensorHolderFromNode(node, 3 + dy1_num + dy2_num, dy3_num);
  return Esphony_1i1dyi_2o2dyo1oOutput{
      builder.GetTensorHolderFromNode(node, 0),
      builder.GetTensorHolderFromNode(node, 1),
      dy1_holders->data(),
      static_cast<int64_t>(dy1_holders->size()),
      dy2_holders->data(),
      static_cast<int64_t>(dy2_holders->size()),
      builder.GetTensorHolderFromNode(node, 2 + dy1_num + dy2_num),
      dy3_holders->data(),
      static_cast<int64_t>(dy3_holders->size()),
  };
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1i1dyi_2o2dyo1o.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1i1dyi_2o2dyo1o_OpCppHeaderGeneration) {
  // 模拟生成的phony_1i1dyi_2o2dyo1o头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_2o2dyo1o_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1i1dyi_2o2dyo1o_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_1i1dyi_2o2dyo1o_c.h"
namespace ge {
namespace es {

struct phony_1i1dyi_2o2dyo1oOutput {
  EsTensorHolder y1;
  EsTensorHolder y2;
  std::vector<EsTensorHolder> dy1;
  std::vector<EsTensorHolder> dy2;
  EsTensorHolder y3;
  std::vector<EsTensorHolder> dy3;
};
/**
 * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:
 *   value
 * @note user needs to provide following inputs for dynamic output numbers:
 *   dy1_num: dynamic output number of dy1
 *   dy2_num: dynamic output number of dy2
 *   dy3_num: dynamic output number of dy3
 */
inline phony_1i1dyi_2o2dyo1oOutput phony_1i1dyi_2o2dyo1o(const EsTensorLike &x, const std::vector<EsTensorHolder> &dx, int64_t dy1_num, int64_t dy2_num, int64_t dy3_num, const std::vector<int64_t> &index={10, 10, 10}, std::unique_ptr<ge::Tensor> value=EsMakeUnique<ge::Tensor>(ge::Tensor())) {
  auto *owner_graph_builder = ResolveBuilder(x, dx);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto esb_dx = TensorsToEsCTensorHolders(dx);
  auto out = Esphony_1i1dyi_2o2dyo1o(x.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), esb_dx.data(), static_cast<int64_t>(esb_dx.size()), dy1_num, dy2_num, dy3_num, index.data(), static_cast<int64_t>(index.size()), static_cast<EsCTensor *>(static_cast<void *>(value.release())));
  std::vector<EsTensorHolder> dy1_dynamic_outs;
  dy1_dynamic_outs.reserve(out.dy1_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy1_num; ++dyn_idx) {
    dy1_dynamic_outs.emplace_back(out.dy1[dyn_idx]);
  }
  std::vector<EsTensorHolder> dy2_dynamic_outs;
  dy2_dynamic_outs.reserve(out.dy2_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy2_num; ++dyn_idx) {
    dy2_dynamic_outs.emplace_back(out.dy2[dyn_idx]);
  }
  std::vector<EsTensorHolder> dy3_dynamic_outs;
  dy3_dynamic_outs.reserve(out.dy3_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.dy3_num; ++dyn_idx) {
    dy3_dynamic_outs.emplace_back(out.dy3[dyn_idx]);
  }
  return {out.y1, out.y2, dy1_dynamic_outs, dy2_dynamic_outs, out.y3, dy3_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1i1dyi_2o2dyo1o.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试多属性情况
TEST_F(GenImplLLT, phony_multi_attr_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_multi_attr应该有以下特征：
  // REG_OP(phony_multi_attr)
  //     .ATTR(li, ListInt, {10, 10, 10})
  //     .ATTR(f, Float, 0.0)
  //     .ATTR(s, String, "s")
  //     .ATTR(b, Bool, true)
  //     .ATTR(lf, ListFloat, {0.1, 0.2})
  //     .ATTR(lb, ListBool, {false, true})
  //     .OP_END_FACTORY_REG(phony_multi_attr);

  // 模拟生成的es_phony_multi_attr_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_multi_attr_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_multi_attr_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

// phony_multi_attr does not produce an output, so the returned EsTensor cannot be used to connect edges.
EsCTensorHolder *Esphony_multi_attr(EsCGraphBuilder *owner_graph_builder, const int64_t *li, int64_t li_num, float f, const char *s, bool b, const float *lf, int64_t lf_num, const bool *lb, int64_t lb_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_multi_attr_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_multi_attr_OpCppGeneration) {
  // 模拟生成的phony_multi_attr.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_multi_attr_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_multi_attr(EsCGraphBuilder *owner_graph_builder, const int64_t *li, int64_t li_num, float f, const char *s, bool b, const float *lf, int64_t lf_num, const bool *lb, int64_t lb_num) {
  ES_ASSERT_NOTNULL(owner_graph_builder);
  auto &builder = *owner_graph_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_multi_attr")
      .Name( builder.GenerateNodeName("phony_multi_attr").GetString())
      .IrDefInputs({
      })
      .IrDefOutputs({
      })
      .IrDefAttrs({
          {
              "li",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListInt",
              ge::es::CreateFromIfNotEqual(std::vector<int64_t>(li, li + li_num), std::vector<int64_t>{10, 10, 10})
          },
          {
              "f",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Float",
              ge::es::CreateFromIfNotEqual(f, static_cast<float>(0.000000))
          },
          {
              "s",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "String",
              ge::es::CreateFromIfNotEqual(ge::AscendString(s), ge::AscendString("s"))
          },
          {
              "b",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Bool",
              ge::es::CreateFromIfNotEqual(b, static_cast<bool>(true))
          },
          {
              "lf",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListFloat",
              ge::es::CreateFromIfNotEqual(std::vector<float>(lf, lf + lf_num), std::vector<float>{0.100000, 0.200000})
          },
          {
              "lb",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListBool",
              ge::es::CreateFromIfNotEqual(std::vector<bool>(lb, lb + lb_num), std::vector<bool>{false, true})
          },
      })
      .Build();

  return builder.GetTensorHolderFromNode(std::move(node), -1);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_multi_attr.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_multi_attr_OpCppHeaderGeneration) {
  // 模拟生成的phony_multi_attr头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_multi_attr_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_multi_attr_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_phony_multi_attr_c.h"
namespace ge {
namespace es {

inline EsTensorHolder phony_multi_attr(const EsGraphBuilder &owner_builder, const std::vector<int64_t> &li={10, 10, 10}, float f=0.000000, const char *s="s", bool b=true, const std::vector<float> &lf={0.100000, 0.200000}, const std::vector<uint8_t> &lb={false, true}) {
  auto out = Esphony_multi_attr(owner_builder.GetCGraphBuilder(), li.data(), static_cast<int64_t>(li.size()), f, s, b, lf.data(), static_cast<int64_t>(lf.size()), static_cast<const bool *>(static_cast<const void *>(lb.data())), static_cast<int64_t>(lb.size()));
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_multi_attr.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试属性补全--必选属性算子
TEST_F(GenImplLLT, phony_req_attrs_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_req_attrs应该有以下特征：
  // REG_OP(phony_req_attrs)
  //     .INPUT(x, TensorType::All())
  //     .OUTPUT(y, TensorType::All())
  //     .REQUIRED_ATTR(req_data_type, Type)
  //     .REQUIRED_ATTR(req_list_data_type, ListType)
  //     .REQUIRED_ATTR(req_list_list_int, ListListInt)
  //     .REQUIRED_ATTR(req_tensor, Tensor)
  //     .REQUIRED_ATTR(req_list_string, ListString)
  //     .OP_END_FACTORY_REG(phony_req_attrs)

  // 模拟生成的es_phony_req_attrs_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_req_attrs_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_req_attrs_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:
 *   req_tensor
 */
EsCTensorHolder *Esphony_req_attrs(EsCTensorHolder *x, C_DataType req_data_type, const C_DataType *req_list_data_type, int64_t req_list_data_type_num, const int64_t **req_list_list_int, int64_t req_list_list_int_num, const int64_t *req_list_list_int_inner_num, EsCTensor *req_tensor, const char **req_list_string, int64_t req_list_string_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_req_attrs_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_req_attrs_OpCppGeneration) {
  // 模拟生成的phony_req_attrs.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_req_attrs_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_req_attrs(EsCTensorHolder *x, C_DataType req_data_type, const C_DataType *req_list_data_type, int64_t req_list_data_type_num, const int64_t **req_list_list_int, int64_t req_list_list_int_num, const int64_t *req_list_list_int_inner_num, EsCTensor *req_tensor, const char **req_list_string, int64_t req_list_string_num) {
  ES_ASSERT_NOTNULL(x);
  auto *owner_builder = ge::es::ResolveBuilder(x);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto req_tensor_stored = builder.AddResource(std::unique_ptr<ge::Tensor>(static_cast<ge::Tensor *>(static_cast<void *>(req_tensor))));
  ES_ASSERT_NOTNULL(req_tensor_stored);
  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_req_attrs")
      .Name( builder.GenerateNodeName("phony_req_attrs").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {
              "req_data_type",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "Type",
              ge::es::CreateFrom(static_cast<ge::DataType>(req_data_type))
          },
          {
              "req_list_data_type",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "ListType",
              ge::es::CreateFrom(std::vector<ge::DataType>([](const C_DataType *es_type_list, int64_t es_type_list_size) { std::vector<ge::DataType> ge_type_list(es_type_list_size); std::transform(es_type_list, es_type_list + es_type_list_size, ge_type_list.begin(), [](C_DataType es_type) { return static_cast<ge::DataType>(es_type);}); return ge_type_list; }(req_list_data_type, req_list_data_type_num)))
          },
          {
              "req_list_list_int",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "ListListInt",
              ge::es::CreateFrom(std::vector<std::vector<int64_t>>([](const int64_t** data, int64_t size, const int64_t *inner_sizes) { std::vector<std::vector<int64_t>> ret; for (int64_t i = 0; i < size; i++) ret.emplace_back(data[i], data[i] + inner_sizes[i]); return ret; } (req_list_list_int, req_list_list_int_num, req_list_list_int_inner_num)))
          },
          {
              "req_tensor",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "Tensor",
              ge::es::CreateFrom(*req_tensor_stored)
          },
          {
              "req_list_string",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "ListString",
              ge::es::CreateFrom(std::vector<ge::AscendString>(req_list_string, req_list_string + req_list_string_num))
          },
      })
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_req_attrs.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_req_attrs_OpCppHeaderGeneration) {
  // 模拟生成的phony_req_attrs头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_req_attrs_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_req_attrs_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_phony_req_attrs_c.h"
namespace ge {
namespace es {

/**
 * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:
 *   req_tensor
 */
inline EsTensorHolder phony_req_attrs(const EsTensorHolder &x, ge::DataType req_data_type, const std::vector<ge::DataType> &req_list_data_type, const std::vector<std::vector<int64_t>> &req_list_list_int, std::unique_ptr<ge::Tensor> req_tensor, const std::vector<const char *> &req_list_string) {
  auto out = Esphony_req_attrs(x.GetCTensorHolder(), static_cast<C_DataType>(req_data_type), DataTypesToEsCDataTypes(req_list_data_type).data(), static_cast<int64_t>(req_list_data_type.size()), ListListTypeToPtrAndCounts<int64_t>(req_list_list_int).first.data(), static_cast<int64_t>(req_list_list_int.size()), ListListTypeToPtrAndCounts<int64_t>(req_list_list_int).second.data(), static_cast<EsCTensor *>(static_cast<void *>(req_tensor.release())), const_cast<const char **>(req_list_string.data()), static_cast<int64_t>(req_list_string.size()));
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_req_attrs.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试属性补全--可算属性算子
TEST_F(GenImplLLT, phony_opt_attrs_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_opt_attrs应该有以下特征：
  // REG_OP(phony_opt_attrs)
  //     .INPUT(x, TensorType::All())
  //     .OUTPUT(y, TensorType::All())
  //     .ATTR(opt_data_type, Type, DT_INT64)
  //     .ATTR(opt_list_data_type, ListType, {DT_FLOAT, DT_DOUBLE})
  //     .ATTR(opt_list_list_int, ListListInt, {{1,2,3}, {3,2,1}})
  //     .ATTR(opt_tensor, Tensor, Tensor())
  //     .ATTR(opt_list_string, ListString, {"test"})
  //     .OP_END_FACTORY_REG(phony_opt_attrs)

  // 模拟生成的es_phony_opt_attrs_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_opt_attrs_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_opt_attrs_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:
 *   opt_tensor
 */
EsCTensorHolder *Esphony_opt_attrs(EsCTensorHolder *x, C_DataType opt_data_type, const C_DataType *opt_list_data_type, int64_t opt_list_data_type_num, const int64_t **opt_list_list_int, int64_t opt_list_list_int_num, const int64_t *opt_list_list_int_inner_num, EsCTensor *opt_tensor, const char **opt_list_string, int64_t opt_list_string_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_opt_attrs_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_opt_attrs_OpCppGeneration) {
  // 模拟生成的phony_opt_attrs.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_opt_attrs_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_opt_attrs(EsCTensorHolder *x, C_DataType opt_data_type, const C_DataType *opt_list_data_type, int64_t opt_list_data_type_num, const int64_t **opt_list_list_int, int64_t opt_list_list_int_num, const int64_t *opt_list_list_int_inner_num, EsCTensor *opt_tensor, const char **opt_list_string, int64_t opt_list_string_num) {
  ES_ASSERT_NOTNULL(x);
  auto *owner_builder = ge::es::ResolveBuilder(x);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto opt_tensor_stored = builder.AddResource(std::unique_ptr<ge::Tensor>(static_cast<ge::Tensor *>(static_cast<void *>(opt_tensor))));
  ES_ASSERT_NOTNULL(opt_tensor_stored);
  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_opt_attrs")
      .Name( builder.GenerateNodeName("phony_opt_attrs").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {
              "opt_data_type",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Type",
              ge::es::CreateFromIfNotEqual(static_cast<ge::DataType>(opt_data_type), ge::DT_INT64)
          },
          {
              "opt_list_data_type",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListType",
              ge::es::CreateFromIfNotEqual(std::vector<ge::DataType>([](const C_DataType *es_type_list, int64_t es_type_list_size) { std::vector<ge::DataType> ge_type_list(es_type_list_size); std::transform(es_type_list, es_type_list + es_type_list_size, ge_type_list.begin(), [](C_DataType es_type) { return static_cast<ge::DataType>(es_type);}); return ge_type_list; }(opt_list_data_type, opt_list_data_type_num)), std::vector<ge::DataType>{ge::DT_FLOAT, ge::DT_DOUBLE})
          },
          {
              "opt_list_list_int",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListListInt",
              ge::es::CreateFromIfNotEqual(std::vector<std::vector<int64_t>>([](const int64_t** data, int64_t size, const int64_t *inner_sizes) { std::vector<std::vector<int64_t>> ret; for (int64_t i = 0; i < size; i++) ret.emplace_back(data[i], data[i] + inner_sizes[i]); return ret; } (opt_list_list_int, opt_list_list_int_num, opt_list_list_int_inner_num)), std::vector<std::vector<int64_t>>{{1, 2, 3}, {3, 2, 1}})
          },
          {
              "opt_tensor",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Tensor",
              ge::es::CreateFrom(*opt_tensor_stored)
          },
          {
              "opt_list_string",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListString",
              ge::es::CreateFromIfNotEqual(std::vector<ge::AscendString>(opt_list_string, opt_list_string + opt_list_string_num), std::vector<ge::AscendString>{"test", "test"})
          },
      })
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_opt_attrs.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_opt_attrs_OpCppHeaderGeneration) {
  // 模拟生成的phony_opt_attrs头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_opt_attrs_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_opt_attrs_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_phony_opt_attrs_c.h"
namespace ge {
namespace es {

/**
 * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:
 *   opt_tensor
 */
inline EsTensorHolder phony_opt_attrs(const EsTensorHolder &x, ge::DataType opt_data_type=ge::DT_INT64, const std::vector<ge::DataType> &opt_list_data_type={ge::DT_FLOAT, ge::DT_DOUBLE}, const std::vector<std::vector<int64_t>> &opt_list_list_int={{1, 2, 3}, {3, 2, 1}}, std::unique_ptr<ge::Tensor> opt_tensor=EsMakeUnique<ge::Tensor>(ge::Tensor()), const std::vector<const char *> &opt_list_string={"test", "test"}) {
  auto out = Esphony_opt_attrs(x.GetCTensorHolder(), static_cast<C_DataType>(opt_data_type), DataTypesToEsCDataTypes(opt_list_data_type).data(), static_cast<int64_t>(opt_list_data_type.size()), ListListTypeToPtrAndCounts<int64_t>(opt_list_list_int).first.data(), static_cast<int64_t>(opt_list_list_int.size()), ListListTypeToPtrAndCounts<int64_t>(opt_list_list_int).second.data(), static_cast<EsCTensor *>(static_cast<void *>(opt_tensor.release())), const_cast<const char **>(opt_list_string.data()), static_cast<int64_t>(opt_list_string.size()));
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_opt_attrs.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试含静态子图算子-IF
TEST_F(GenImplLLT, phony_If_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_If应该有以下特征：
  // REG_OP(phony_If)
  //     .INPUT(cond, TensorType::ALL())
  //     .DYNAMIC_INPUT(input, TensorType::ALL())
  //     .DYNAMIC_OUTPUT(output, TensorType::ALL())
  //     .GRAPH(then_branch)
  //     .GRAPH(else_branch)
  //     .OP_END_FACTORY_REG(phony_If)

  // 模拟生成的es_phony_If_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_If_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_If_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **output;
  int64_t output_num;
} Esphony_IfOutput;
/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   then_branch
 *   else_branch
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
Esphony_IfOutput Esphony_If(EsCTensorHolder *cond, EsCTensorHolder **input, int64_t input_num, int64_t output_num, EsCGraph *then_branch, EsCGraph *else_branch);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_If_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_If_OpCppHeaderGeneration) {
  // 模拟生成的phony_If头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_If_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_If_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_If_c.h"
namespace ge {
namespace es {

/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   then_branch
 *   else_branch
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
inline std::vector<EsTensorHolder> phony_If(const EsTensorLike &cond, const std::vector<EsTensorHolder> &input, int64_t output_num, std::unique_ptr<ge::Graph> then_branch, std::unique_ptr<ge::Graph> else_branch) {
  auto *owner_graph_builder = ResolveBuilder(cond, input);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto esb_input = TensorsToEsCTensorHolders(input);
  auto out = Esphony_If(cond.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), esb_input.data(), static_cast<int64_t>(esb_input.size()), output_num, static_cast<EsCGraph *>(static_cast<void *>(then_branch.release())), static_cast<EsCGraph *>(static_cast<void *>(else_branch.release())));
  std::vector<EsTensorHolder> output_dynamic_outs;
  output_dynamic_outs.reserve(out.output_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.output_num; ++dyn_idx) {
    output_dynamic_outs.emplace_back(out.output[dyn_idx]);
  }
  return {output_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_If.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试含动态子图算子-Case
TEST_F(GenImplLLT, phony_Case_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_Case应该有以下特征：
  // REG_OP(phony_Case)
  //     .INPUT(branch_index, DT_INT32)
  //     .DYNAMIC_INPUT(input, TensorType::ALL())
  //     .DYNAMIC_OUTPUT(output, TensorType::ALL())
  //     .DYNAMIC_GRAPH(branches)
  //     .OP_END_FACTORY_REG(phony_Case)

  // 模拟生成的es_phony_Case_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_Case_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_Case_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **output;
  int64_t output_num;
} Esphony_CaseOutput;
/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   branches
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
Esphony_CaseOutput Esphony_Case(EsCTensorHolder *branch_index, EsCTensorHolder **input, int64_t input_num, int64_t output_num, EsCGraph **branches, int64_t branches_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_Case_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_Case_OpCppHeaderGeneration) {
  // 模拟生成的phony_Case头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_Case_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_Case_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_Case_c.h"
namespace ge {
namespace es {

/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   branches
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
inline std::vector<EsTensorHolder> phony_Case(const EsTensorLike &branch_index, const std::vector<EsTensorHolder> &input, int64_t output_num, std::vector<std::unique_ptr<ge::Graph>> branches) {
  auto *owner_graph_builder = ResolveBuilder(branch_index, input);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto esb_input = TensorsToEsCTensorHolders(input);
  auto esb_branches= static_cast<int64_t>(branches.size());
  auto out = Esphony_Case(branch_index.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), esb_input.data(), static_cast<int64_t>(esb_input.size()), output_num, GeGraphsToEsCGraphs(std::move(branches)).data(), esb_branches);
  std::vector<EsTensorHolder> output_dynamic_outs;
  output_dynamic_outs.reserve(out.output_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.output_num; ++dyn_idx) {
    output_dynamic_outs.emplace_back(out.output[dyn_idx]);
  }
  return {output_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_Case.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试含子图算子-PartitionedCall
TEST_F(GenImplLLT, phony_PartitionedCall_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_PartitionedCall应该有以下特征：
  // REG_OP(phony_PartitionedCall)
  //     .DYNAMIC_INPUT(args, TensorType::ALL())
  //     .DYNAMIC_OUTPUT(output, TensorType::ALL())
  //     .GRAPH(f)
  //     .ATTR(config, String, "")
  //     .ATTR(config_proto, String, "")
  //     .ATTR(executor_type, String, "")
  //     .OP_END_FACTORY_REG(phony_PartitionedCall)

  // 模拟生成的es_phony_PartitionedCall_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_PartitionedCall_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_PartitionedCall_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **output;
  int64_t output_num;
} Esphony_PartitionedCallOutput;
/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   f
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
Esphony_PartitionedCallOutput Esphony_PartitionedCall(EsCTensorHolder **args, int64_t args_num, int64_t output_num, EsCGraph *f, const char *config, const char *config_proto, const char *executor_type);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_PartitionedCall_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_PartitionedCall_OpCppHeaderGeneration) {
  // 模拟生成的phony_PartitionedCall头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_PartitionedCall_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_PartitionedCall_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_phony_PartitionedCall_c.h"
namespace ge {
namespace es {

/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   f
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
inline std::vector<EsTensorHolder> phony_PartitionedCall(const std::vector<EsTensorHolder> &args, int64_t output_num, std::unique_ptr<ge::Graph> f, const char *config="", const char *config_proto="", const char *executor_type="") {
  auto esb_args = TensorsToEsCTensorHolders(args);
  auto out = Esphony_PartitionedCall(esb_args.data(), static_cast<int64_t>(esb_args.size()), output_num, static_cast<EsCGraph *>(static_cast<void *>(f.release())), config, config_proto, executor_type);
  std::vector<EsTensorHolder> output_dynamic_outs;
  output_dynamic_outs.reserve(out.output_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.output_num; ++dyn_idx) {
    output_dynamic_outs.emplace_back(out.output[dyn_idx]);
  }
  return {output_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_PartitionedCall.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试需特殊处理的含子图算子-需要处理cond子图-While
TEST_F(GenImplLLT, While_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，While应该有以下特征：
  // REG_OP(While)
  //     .DYNAMIC_INPUT(input, TensorType::ALL())
  //     .DYNAMIC_OUTPUT(output, TensorType::ALL())
  //     .GRAPH(cond)
  //     .GRAPH(body)
  //     .ATTR(parallel_iterations, Int, 10)
  //     .OP_END_FACTORY_REG(While)

  // 模拟生成的es_While_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_While_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_While_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **output;
  int64_t output_num;
} EsWhileOutput;
/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   cond
 *   body
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
EsWhileOutput EsWhile(EsCTensorHolder **input, int64_t input_num, int64_t output_num, EsCGraph *cond, EsCGraph *body, int64_t parallel_iterations);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_While_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, While_OpCppGeneration) {
  // 模拟生成的While.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_While_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsWhileOutput EsWhile(EsCTensorHolder **input, int64_t input_num, int64_t output_num, EsCGraph *cond, EsCGraph *body, int64_t parallel_iterations) {
  ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(input_num));
  auto *owner_builder = ge::es::ResolveBuilder(std::vector<EsCTensorHolder *>(input, input + input_num));
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  ge::Graph* input_subgraph;
  input_subgraph = static_cast<ge::Graph *>(static_cast<void *>(cond));
  auto cond_node_type = ge::AscendString();
  int net_output_num_of_cond = 0;
  std::unordered_set<int64_t> cond_data_indexes;
  cond_data_indexes.reserve(input_num);
  for (auto &sub_graph_node : input_subgraph->GetDirectNode()) {
    ES_ASSERT_GRAPH_SUCCESS(sub_graph_node.GetType(cond_node_type));
    if (cond_node_type != "Data") {
      if (cond_node_type == "NetOutput") {
        ES_ASSERT_TRUE(net_output_num_of_cond <= 1);
        net_output_num_of_cond++;
        ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(output_num));
      }
      continue;
    }
    int64_t index_value;
    ES_ASSERT_GRAPH_SUCCESS(sub_graph_node.GetAttr("index", index_value));
    ES_ASSERT_TRUE(cond_data_indexes.insert(index_value).second);
    ES_ASSERT_TRUE(index_value < input_num);
  }
  ES_ASSERT_TRUE(net_output_num_of_cond == 1);
  auto cond_ptr = builder.AddResource(std::unique_ptr<ge::Graph>(input_subgraph));

  input_subgraph = static_cast<ge::Graph *>(static_cast<void *>(body));
  auto body_node_type = ge::AscendString();
  int net_output_num_of_body = 0;
  std::unordered_set<int64_t> body_data_indexes;
  body_data_indexes.reserve(input_num);
  for (auto &sub_graph_node : input_subgraph->GetDirectNode()) {
    ES_ASSERT_GRAPH_SUCCESS(sub_graph_node.GetType(body_node_type));
    if (body_node_type != "Data") {
      if (body_node_type == "NetOutput") {
        ES_ASSERT_TRUE(net_output_num_of_body <= 1);
        net_output_num_of_body++;
        ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(output_num));
        auto subgraph_output_cnt = sub_graph_node.GetInputsSize();
          ES_ASSERT_TRUE(static_cast<int64_t>(subgraph_output_cnt)  == output_num);
      }
      continue;
    }
    int64_t index_value;
    ES_ASSERT_GRAPH_SUCCESS(sub_graph_node.GetAttr("index", index_value));
    ES_ASSERT_TRUE(body_data_indexes.insert(index_value).second);
    ES_ASSERT_TRUE(index_value < input_num);
  }
  ES_ASSERT_TRUE(net_output_num_of_body == 1);
  auto body_ptr = builder.AddResource(std::unique_ptr<ge::Graph>(input_subgraph));

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("While")
      .Name( builder.GenerateNodeName("While").GetString())
      .IrDefInputs({
          {"input", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
      })
      .IrDefOutputs({
          {"output", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
      })
      .IrDefAttrs({
          {
              "parallel_iterations",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Int",
              ge::es::CreateFromIfNotEqual(parallel_iterations, static_cast<int64_t>(10))
          },
      })
      .InstanceDynamicInputNum("input", static_cast<int32_t>(input_num))
      .InstanceDynamicOutputNum("output", static_cast<int32_t>(output_num))
      .Build();

  if ((input != nullptr) && (input_num > 0)) {
    for (int64_t i = 0; i < input_num; ++i) {
      auto one_input = input[i];
      ES_ASSERT_NOTNULL(one_input);
      ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, one_input->GetProducer(), one_input->GetOutIndex(), node, 0 + i));
    }
  }

  node.SetSubgraph("cond", *cond_ptr);
  node.SetSubgraph("body", *body_ptr);

  auto output_holders = builder.CreateDynamicTensorHolderFromNode(node, 0, output_num);
  return EsWhileOutput{
      output_holders->data(),
      static_cast<int64_t>(output_holders->size()),
  };
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_While.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, While_OpCppHeaderGeneration) {
  // 模拟生成的While头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_While_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_While_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_While_c.h"
namespace ge {
namespace es {

/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   cond
 *   body
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
inline std::vector<EsTensorHolder> While(const std::vector<EsTensorHolder> &input, int64_t output_num, std::unique_ptr<ge::Graph> cond, std::unique_ptr<ge::Graph> body, int64_t parallel_iterations=10) {
  auto esb_input = TensorsToEsCTensorHolders(input);
  auto out = EsWhile(esb_input.data(), static_cast<int64_t>(esb_input.size()), output_num, static_cast<EsCGraph *>(static_cast<void *>(cond.release())), static_cast<EsCGraph *>(static_cast<void *>(body.release())), parallel_iterations);
  std::vector<EsTensorHolder> output_dynamic_outs;
  output_dynamic_outs.reserve(out.output_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.output_num; ++dyn_idx) {
    output_dynamic_outs.emplace_back(out.output[dyn_idx]);
  }
  return {output_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_While.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试含混合子图算子
TEST_F(GenImplLLT, phony_mix_subgraphs_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_mix_subgraphs应该有以下特征：
  // REG_OP(phony_mix_subgraphs)
  //     .OPTIONAL_INPUT(opt_input, TensorType::All())
  //     .DYNAMIC_INPUT(input, TensorType::ALL())
  //     .DYNAMIC_OUTPUT(output, TensorType::ALL())
  //     .GRAPH(cond)
  //     .DYNAMIC_GRAPH(branches)
  //     .OP_END_FACTORY_REG(phony_mix_subgraphs)

  // 模拟生成的es_phony_mix_subgraphs_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_mix_subgraphs_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_mix_subgraphs_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **output;
  int64_t output_num;
} Esphony_mix_subgraphsOutput;
/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   cond
 *   branches
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
Esphony_mix_subgraphsOutput Esphony_mix_subgraphs(EsCTensorHolder *opt_input, EsCTensorHolder **input, int64_t input_num, int64_t output_num, EsCGraph *cond, EsCGraph **branches, int64_t branches_num);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_mix_subgraphs_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_mix_subgraphs_OpCppGeneration) {
  // 模拟生成的phony_mix_subgraphs.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_mix_subgraphs_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
Esphony_mix_subgraphsOutput Esphony_mix_subgraphs(EsCTensorHolder *opt_input, EsCTensorHolder **input, int64_t input_num, int64_t output_num, EsCGraph *cond, EsCGraph **branches, int64_t branches_num) {
  ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(input_num));
  auto *owner_builder = ge::es::ResolveBuilder(opt_input, std::vector<EsCTensorHolder *>(input, input + input_num));
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  ge::Graph* input_subgraph;
  input_subgraph = static_cast<ge::Graph *>(static_cast<void *>(cond));
  auto cond_node_type = ge::AscendString();
  int net_output_num_of_cond = 0;
  std::unordered_set<int64_t> cond_data_indexes;
  cond_data_indexes.reserve(input_num);
  for (auto &sub_graph_node : input_subgraph->GetDirectNode()) {
    ES_ASSERT_GRAPH_SUCCESS(sub_graph_node.GetType(cond_node_type));
    if (cond_node_type != "Data") {
      if (cond_node_type == "NetOutput") {
        ES_ASSERT_TRUE(net_output_num_of_cond <= 1);
        net_output_num_of_cond++;
        ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(output_num));
        auto subgraph_output_cnt = sub_graph_node.GetInputsSize();
          ES_ASSERT_TRUE(static_cast<int64_t>(subgraph_output_cnt)  == output_num);
      }
      continue;
    }
    int64_t index_value;
    ES_ASSERT_GRAPH_SUCCESS(sub_graph_node.GetAttr("index", index_value));
    ES_ASSERT_TRUE(cond_data_indexes.insert(index_value).second);
    ES_ASSERT_TRUE(index_value < input_num);
  }
  ES_ASSERT_TRUE(net_output_num_of_cond == 1);
  auto cond_ptr = builder.AddResource(std::unique_ptr<ge::Graph>(input_subgraph));

  std::vector<ge::Graph> dynamic_branches;
  for (int64_t subgraph_idx = 0; subgraph_idx < branches_num; subgraph_idx++) {
    const auto subgraph_instance = branches[subgraph_idx];
    input_subgraph = static_cast<ge::Graph *>(static_cast<void *>(subgraph_instance));
    auto branches_node_type = ge::AscendString();
    int net_output_num_of_branches = 0;
    std::unordered_set<int64_t> branches_data_indexes;
    branches_data_indexes.reserve(input_num);
    for (auto &sub_graph_node : input_subgraph->GetDirectNode()) {
      ES_ASSERT_GRAPH_SUCCESS(sub_graph_node.GetType(branches_node_type));
      if (branches_node_type != "Data") {
        if (branches_node_type == "NetOutput") {
          ES_ASSERT_TRUE(net_output_num_of_branches <= 1);
          net_output_num_of_branches++;
          ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(output_num));
          auto subgraph_output_cnt = sub_graph_node.GetInputsSize();
            ES_ASSERT_TRUE(static_cast<int64_t>(subgraph_output_cnt)  == output_num);
        }
        continue;
      }
      int64_t index_value;
      ES_ASSERT_GRAPH_SUCCESS(sub_graph_node.GetAttr("index", index_value));
      ES_ASSERT_TRUE(branches_data_indexes.insert(index_value).second);
      ES_ASSERT_TRUE(index_value < input_num);
    }
    ES_ASSERT_TRUE(net_output_num_of_branches == 1);
    dynamic_branches.emplace_back(*builder.AddResource(std::unique_ptr<ge::Graph>(input_subgraph)));
  }

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_mix_subgraphs")
      .Name( builder.GenerateNodeName("phony_mix_subgraphs").GetString())
      .IrDefInputs({
          {"opt_input", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
          {"input", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
      })
      .IrDefOutputs({
          {"output", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
      })
      .IrDefAttrs({
      })
      .InstanceDynamicInputNum("input", static_cast<int32_t>(input_num))
      .InstanceDynamicOutputNum("output", static_cast<int32_t>(output_num))
      .Build();

  if (opt_input != nullptr) {
    ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, opt_input->GetProducer(), opt_input->GetOutIndex(), node, 0));
  }
  if ((input != nullptr) && (input_num > 0)) {
    for (int64_t i = 0; i < input_num; ++i) {
      auto one_input = input[i];
      ES_ASSERT_NOTNULL(one_input);
      ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, one_input->GetProducer(), one_input->GetOutIndex(), node, 1 + i));
    }
  }

  node.SetSubgraph("cond", *cond_ptr);
  node.SetSubgraphs("branches", dynamic_branches);

  auto output_holders = builder.CreateDynamicTensorHolderFromNode(node, 0, output_num);
  return Esphony_mix_subgraphsOutput{
      output_holders->data(),
      static_cast<int64_t>(output_holders->size()),
  };
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_mix_subgraphs.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_mix_subgraphs_OpCppHeaderGeneration) {
  // 模拟生成的phony_mix_subgraphs头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_mix_subgraphs_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_mix_subgraphs_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_mix_subgraphs_c.h"
namespace ge {
namespace es {

/**
 * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:
 *   cond
 *   branches
 * @note user needs to provide following inputs for dynamic output numbers:
 *   output_num: dynamic output number of output
 */
inline std::vector<EsTensorHolder> phony_mix_subgraphs(const EsTensorLike &opt_input, const std::vector<EsTensorHolder> &input, int64_t output_num, std::unique_ptr<ge::Graph> cond, std::vector<std::unique_ptr<ge::Graph>> branches) {
  auto *owner_graph_builder = ResolveBuilder(opt_input, input);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto esb_input = TensorsToEsCTensorHolders(input);
  auto esb_branches= static_cast<int64_t>(branches.size());
  auto out = Esphony_mix_subgraphs(opt_input.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), esb_input.data(), static_cast<int64_t>(esb_input.size()), output_num, static_cast<EsCGraph *>(static_cast<void *>(cond.release())), GeGraphsToEsCGraphs(std::move(branches)).data(), esb_branches);
  std::vector<EsTensorHolder> output_dynamic_outs;
  output_dynamic_outs.reserve(out.output_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.output_num; ++dyn_idx) {
    output_dynamic_outs.emplace_back(out.output[dyn_idx]);
  }
  return {output_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_mix_subgraphs.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();
  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试含同名输出输出与属性的算子
TEST_F(GenImplLLT, phony_dup_name_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_dup_name应该有以下特征：
  // REG_OP(phony_dup_name)
  //     .INPUT(x, TensorType::ALL())
  //     .DYNAMIC_INPUT(dx, TensorType::All())
  //     .DYNAMIC_OUTPUT(x, TensorType::ALL())
  //     .DYNAMIC_OUTPUT(dx, TensorType::All())
  //     .ATTR(index, Int, 0)
  //     .ATTR(value, Tensor, Tensor())
  //     .ATTR(x, Int, 1)
  //     .ATTR(dx, Int, 2)
  //     .OP_END_FACTORY_REG(phony_dup_name);

  // 模拟生成的es_phony_dup_name_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_dup_name_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_dup_name_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **ref_x;
  int64_t ref_x_num;
  EsCTensorHolder **ref_dx;
  int64_t ref_dx_num;
} Esphony_dup_nameOutput;
/**
 * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:
 *   value
 * @note user needs to provide following inputs for dynamic output numbers:
 *   ref_x_num: dynamic output number of x
 *   ref_dx_num: dynamic output number of dx
 */
Esphony_dup_nameOutput Esphony_dup_name(EsCTensorHolder *x, EsCTensorHolder **dx, int64_t dx_num, int64_t ref_x_num, int64_t ref_dx_num, int64_t index, EsCTensor *value, int64_t attr_x, int64_t attr_dx);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_dup_name_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_dup_name_OpCppGeneration) {
  // 模拟生成的phony_dup_name.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_dup_name_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
Esphony_dup_nameOutput Esphony_dup_name(EsCTensorHolder *x, EsCTensorHolder **dx, int64_t dx_num, int64_t ref_x_num, int64_t ref_dx_num, int64_t index, EsCTensor *value, int64_t attr_x, int64_t attr_dx) {
  ES_ASSERT_NOTNULL(x);
  ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(dx_num));
  auto *owner_builder = ge::es::ResolveBuilder(x, std::vector<EsCTensorHolder *>(dx, dx + dx_num));
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto value_stored = builder.AddResource(std::unique_ptr<ge::Tensor>(static_cast<ge::Tensor *>(static_cast<void *>(value))));
  ES_ASSERT_NOTNULL(value_stored);
  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_dup_name")
      .Name( builder.GenerateNodeName("phony_dup_name").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"dx", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
      })
      .IrDefOutputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
          {"dx", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
      })
      .IrDefAttrs({
          {
              "index",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Int",
              ge::es::CreateFromIfNotEqual(index, static_cast<int64_t>(0))
          },
          {
              "value",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Tensor",
              ge::es::CreateFrom(*value_stored)
          },
          {
              "x",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Int",
              ge::es::CreateFromIfNotEqual(attr_x, static_cast<int64_t>(1))
          },
          {
              "dx",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Int",
              ge::es::CreateFromIfNotEqual(attr_dx, static_cast<int64_t>(2))
          },
      })
      .InstanceDynamicInputNum("dx", static_cast<int32_t>(dx_num))
      .InstanceDynamicOutputNum("x", static_cast<int32_t>(ref_x_num))
      .InstanceDynamicOutputNum("dx", static_cast<int32_t>(ref_dx_num))
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  if ((dx != nullptr) && (dx_num > 0)) {
    for (int64_t i = 0; i < dx_num; ++i) {
      auto one_dx = dx[i];
      ES_ASSERT_NOTNULL(one_dx);
      ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, one_dx->GetProducer(), one_dx->GetOutIndex(), node, 1 + i));
    }
  }
  auto ref_x_holders = builder.CreateDynamicTensorHolderFromNode(node, 0, ref_x_num);
  auto ref_dx_holders = builder.CreateDynamicTensorHolderFromNode(node, 0 + ref_x_num, ref_dx_num);
  return Esphony_dup_nameOutput{
      ref_x_holders->data(),
      static_cast<int64_t>(ref_x_holders->size()),
      ref_dx_holders->data(),
      static_cast<int64_t>(ref_dx_holders->size()),
  };
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_dup_name.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_dup_name_OpCppHeaderGeneration) {
  // 模拟生成的phony_dup_name头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_dup_name_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_dup_name_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_dup_name_c.h"
namespace ge {
namespace es {

struct phony_dup_nameOutput {
  std::vector<EsTensorHolder> ref_x;
  std::vector<EsTensorHolder> ref_dx;
};
/**
 * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:
 *   value
 * @note user needs to provide following inputs for dynamic output numbers:
 *   ref_x_num: dynamic output number of x
 *   ref_dx_num: dynamic output number of dx
 */
inline phony_dup_nameOutput phony_dup_name(const EsTensorLike &x, const std::vector<EsTensorHolder> &dx, int64_t ref_x_num, int64_t ref_dx_num, int64_t index=0, std::unique_ptr<ge::Tensor> value=EsMakeUnique<ge::Tensor>(ge::Tensor()), int64_t attr_x=1, int64_t attr_dx=2) {
  auto *owner_graph_builder = ResolveBuilder(x, dx);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto esb_dx = TensorsToEsCTensorHolders(dx);
  auto out = Esphony_dup_name(x.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), esb_dx.data(), static_cast<int64_t>(esb_dx.size()), ref_x_num, ref_dx_num, index, static_cast<EsCTensor *>(static_cast<void *>(value.release())), attr_x, attr_dx);
  std::vector<EsTensorHolder> ref_x_dynamic_outs;
  ref_x_dynamic_outs.reserve(out.ref_x_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.ref_x_num; ++dyn_idx) {
    ref_x_dynamic_outs.emplace_back(out.ref_x[dyn_idx]);
  }
  std::vector<EsTensorHolder> ref_dx_dynamic_outs;
  ref_dx_dynamic_outs.reserve(out.ref_dx_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.ref_dx_num; ++dyn_idx) {
    ref_dx_dynamic_outs.emplace_back(out.ref_dx[dyn_idx]);
  }
  return {ref_x_dynamic_outs, ref_dx_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_dup_name.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试只有全都是可选输入的算子
TEST_F(GenImplLLT, phony_3opi_1o_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_3opi_1o应该有以下特征：
  // REG_OP(phony_3opi_1o)
  //   .OPTIONAL_INPUT(x1, TensorType::All())
  //   .OPTIONAL_INPUT(x2, TensorType::All())
  //   .OPTIONAL_INPUT(x3, TensorType::All())
  //   .OUTPUT(y, TensorType::All())
  //   .OP_END_FACTORY_REG(phony_3opi_1o);
  // 模拟生成的es_phony_3opi_1o_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_3opi_1o_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_3opi_1o_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_3opi_1o(EsCTensorHolder *x1, EsCTensorHolder *x2, EsCTensorHolder *x3, EsCGraphBuilder *owner_graph_builder);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_3opi_1o_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_3opi_1o_OpCppGeneration) {
  // 模拟生成的phony_3opi_1o.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_3opi_1o_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_3opi_1o(EsCTensorHolder *x1, EsCTensorHolder *x2, EsCTensorHolder *x3, EsCGraphBuilder *owner_graph_builder) {
  auto *owner_builder = ge::es::ResolveBuilder(x1, x2, x3, owner_graph_builder);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_3opi_1o")
      .Name( builder.GenerateNodeName("phony_3opi_1o").GetString())
      .IrDefInputs({
          {"x1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
          {"x2", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
          {"x3", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
      })
      .Build();

  if (x1 != nullptr) {
    ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x1->GetProducer(), x1->GetOutIndex(), node, 0));
  }
  if (x2 != nullptr) {
    ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x2->GetProducer(), x2->GetOutIndex(), node, 1));
  }
  if (x3 != nullptr) {
    ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x3->GetProducer(), x3->GetOutIndex(), node, 2));
  }
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_3opi_1o.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_3opi_1o_OpCppHeaderGeneration) {
  // 模拟生成的phony_3opi_1o头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_3opi_1o_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_3opi_1o_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_3opi_1o_c.h"
namespace ge {
namespace es {

/**
 * @note at least one of the following input arguments should be EsTensorHolder object or owner_builder should be provided:
 *   x1
 *   x2
 *   x3
 */
inline EsTensorHolder phony_3opi_1o(const EsTensorLike &x1=nullptr, const EsTensorLike &x2=nullptr, const EsTensorLike &x3=nullptr, const EsGraphBuilder *owner_builder = nullptr) {
  auto *owner_graph_builder = ResolveBuilder(x1, x2, x3, owner_builder);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto out = Esphony_3opi_1o(x1.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), x2.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), x3.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), owner_builder == nullptr ? nullptr : owner_builder->GetCGraphBuilder());
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_3opi_1o.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

// 测试只有一个可选输入的算子
TEST_F(GenImplLLT, phony_1opi_1o_OpGeneration) {
  // 根据stub_geir_ops.cc中的定义，phony_1opi_1o应该有以下特征：
  // REG_OP(phony_1opi_1o)
  //     .OPTIONAL_INPUT(x1, TensorType::All())
  //     .OUTPUT(y, TensorType::All())
  //     .ATTR(dt, Type, DT_FLOAT)
  //     .OP_END_FACTORY_REG(phony_1opi_1o);
  // 模拟生成的es_phony_1opi_1o_c.h文件内容
  std::string expected_header_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1opi_1o_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1opi_1o_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_1opi_1o(EsCTensorHolder *x1, EsCGraphBuilder *owner_graph_builder, bool flag);
#ifdef __cplusplus
}
#endif
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1opi_1o_c.h");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_header_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1opi_1o_OpCppGeneration) {
  // 模拟生成的phony_1opi_1o.cpp文件内容
  std::string expected_cc_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#include "es_phony_1opi_1o_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"
#include "es_tensor_like.h"
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
EsCTensorHolder *Esphony_1opi_1o(EsCTensorHolder *x1, EsCGraphBuilder *owner_graph_builder, bool flag) {
  auto *owner_builder = ge::es::ResolveBuilder(x1, owner_graph_builder);
  ES_ASSERT_NOTNULL(owner_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_graph_builder is provided when supported.");
  auto &builder = *owner_builder;
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("phony_1opi_1o")
      .Name( builder.GenerateNodeName("phony_1opi_1o").GetString())
      .IrDefInputs({
          {"x1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {
              "flag",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "Bool",
              ge::es::CreateFrom(static_cast<bool>(flag))
          },
      })
      .Build();

  if (x1 != nullptr) {
    ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x1->GetProducer(), x1->GetOutIndex(), node, 0));
  }
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif

)";

  // 验证头文件内容 - 直接比较内容是否一致
  std::ifstream read_header(test_output_dir_ + "es_phony_1opi_1o.cpp");
  std::stringstream header_content;
  header_content << read_header.rdbuf();
  read_header.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(header_content.str()), Normalize(expected_cc_content))
      << "Generated header content does not match expected content";
}

TEST_F(GenImplLLT, phony_1opi_1o_OpCppHeaderGeneration) {
  // 模拟生成的phony_1opi_1o头文件内容
  std::string expected_cpp_content = R"(
/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

/*********************************************************************************************************************
 This file is GENERATED by bin/gen_esb, do not edit it manually
*********************************************************************************************************************/

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1opi_1o_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_OP_phony_1opi_1o_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "es_tensor_like.h"
#include "es_log.h"
#include <iostream>
#include "es_phony_1opi_1o_c.h"
namespace ge {
namespace es {

/**
 * @note at least one of the following input arguments should be EsTensorHolder object or owner_builder should be provided:
 *   x1
 */
inline EsTensorHolder phony_1opi_1o(const EsTensorLike &x1, const EsGraphBuilder *owner_builder, bool flag) {
  auto *owner_graph_builder = ResolveBuilder(x1, owner_builder);
  ES_ASSERT_NOTNULL(owner_graph_builder, "Failed to resolve owner builder: please ensure at least one input tensor or an explicit owner_builder is provided when supported.");
  auto out = Esphony_1opi_1o(x1.ToTensorHolder(owner_graph_builder).GetCTensorHolder(), owner_builder == nullptr ? nullptr : owner_builder->GetCGraphBuilder(), flag);
  return out;
}
}  // namespace es
}  // namespace ge
#endif

)";

  // 验证C++头文件内容 - 直接比较内容是否一致
  std::ifstream read_cpp(test_output_dir_ + "es_phony_1opi_1o.h");
  std::stringstream cpp_content;
  cpp_content << read_cpp.rdbuf();
  read_cpp.close();

  // 比较生成的文件内容与预期内容是否完全一致
  EXPECT_EQ(Normalize(cpp_content.str()), Normalize(expected_cpp_content)) << "Generated C++ header content does not "
                                                                              "match "
                                                                              "expected content";
}

TEST_F(GenImplLLTExtractHistory, ExtractHistoryGeneratesFiles) {
  opts.release_version = "8.0.RC1";
  opts.release_date = "2024-09-30";
  opts.branch_name = "master";

  ge::es::GenEsImpl(opts);

  const std::string index_path = opts.output_dir + "index.json";
  const std::string metadata_path = opts.output_dir + "registry/8.0.RC1/metadata.json";
  const std::string operators_path = opts.output_dir + "registry/8.0.RC1/operators.json";

  EXPECT_TRUE(ge::IsFile(index_path.c_str()));
  EXPECT_TRUE(ge::IsFile(metadata_path.c_str()));
  EXPECT_TRUE(ge::IsFile(operators_path.c_str()));

  nlohmann::json index_json;
  std::string error_msg;
  ASSERT_TRUE(ge::es::history::ReadJsonFile(index_path, index_json, error_msg));
  ASSERT_TRUE(index_json.contains("version"));
  EXPECT_EQ(index_json["version"], "1.0.0");
  ASSERT_TRUE(index_json.contains("releases"));
  ASSERT_TRUE(index_json["releases"].is_array());
  ASSERT_EQ(index_json["releases"].size(), 1U);
  EXPECT_EQ(index_json["releases"][0]["release_version"], "8.0.RC1");
  EXPECT_EQ(index_json["releases"][0]["release_date"], "2024-09-30");

  nlohmann::json meta_json;
  ASSERT_TRUE(ge::es::history::ReadJsonFile(metadata_path, meta_json, error_msg));
  EXPECT_EQ(meta_json["release_version"], "8.0.RC1");
  EXPECT_EQ(meta_json["branch_name"], "master");

  nlohmann::json ops_json;
  ASSERT_TRUE(ge::es::history::ReadJsonFile(operators_path, ops_json, error_msg));
  ASSERT_TRUE(ops_json.contains("operators"));
  ASSERT_TRUE(ops_json["operators"].is_array());
  EXPECT_GT(ops_json["operators"].size(), 0U);
}

TEST_F(GenImplLLTExtractHistory, ExtractHistoryWithEmptyReleaseDate) {
  opts.release_version = "8.0.RC2";
  opts.branch_name = "develop";

  ge::es::GenEsImpl(opts);

  const std::string index_path = opts.output_dir + "index.json";
  nlohmann::json index_json;
  std::string error_msg;
  ASSERT_TRUE(ge::es::history::ReadJsonFile(index_path, index_json, error_msg));
  ASSERT_EQ(index_json["releases"].size(), 1U);
  EXPECT_EQ(index_json["releases"][0]["release_version"], "8.0.RC2");

  std::string date = index_json["releases"][0]["release_date"].get<std::string>();
  EXPECT_TRUE(ge::es::history::ValidateReleaseDateFormat(date));
}

TEST_F(GenImplLLTExtractHistory, ExtractHistoryUpdatesIndex) {
  opts.release_version = "8.0.RC1";
  opts.release_date = "2024-09-30";
  opts.branch_name = "master";

  ge::es::GenEsImpl(opts);

  const std::string index_path = opts.output_dir + "index.json";
  nlohmann::json index_json;
  std::string error_msg;
  ASSERT_TRUE(ge::es::history::ReadJsonFile(index_path, index_json, error_msg));
  ASSERT_EQ(index_json["releases"].size(), 1U);

  opts.release_version = "8.0.RC2";
  opts.release_date = "2024-10-15";
  ge::es::GenEsImpl(opts);

  ASSERT_TRUE(ge::es::history::ReadJsonFile(index_path, index_json, error_msg));
  ASSERT_EQ(index_json["releases"].size(), 2U);
  EXPECT_EQ(index_json["releases"][0]["release_version"], "8.0.RC1");
  EXPECT_EQ(index_json["releases"][1]["release_version"], "8.0.RC2");

  const std::string metadata_path_1 = opts.output_dir + "registry/8.0.RC1/metadata.json";
  const std::string metadata_path_2 = opts.output_dir + "registry/8.0.RC2/metadata.json";
  EXPECT_TRUE(ge::IsFile(metadata_path_1.c_str()));
  EXPECT_TRUE(ge::IsFile(metadata_path_2.c_str()));
}

TEST_F(GenImplLLTExtractHistory, ExtractHistoryThrowsOnEmptyReleaseVersion) {
  opts.release_version = "";
  opts.release_date = "2024-09-30";
  opts.branch_name = "master";

  ExpectInvalidArgumentErrorContains([&]() {
    ge::es::GenEsImpl(opts);
  }, "release_version");

  ExpectInvalidArgumentErrorContains([&]() {
    ge::es::GenEsImpl(opts);
  }, "The required parameter release_version for history registry generator is not set.");
}

TEST_F(GenImplLLTExtractHistory, ExtractHistoryThrowsOnInvalidReleaseDate) {
  opts.release_version = "8.0.RC1";
  opts.release_date = "20240930";
  opts.branch_name = "master";

  ExpectInvalidArgumentErrorContains([&]() {
    ge::es::GenEsImpl(opts);
  }, "Given release_date parameter for history registry generator is not in the correct format (YYYY-MM-DD).");
}

TEST_F(GenImplLLTExtractHistory, ExtractHistoryThrowsOnDuplicateReleaseVersion) {
  opts.release_version = "8.0.RC1";
  opts.release_date = "2024-09-30";
  opts.branch_name = "master";

  ge::es::GenEsImpl(opts);

  opts.release_date = "2024-10-01";
  ExpectInvalidArgumentErrorContains([&]() {
    ge::es::GenEsImpl(opts);
  }, "Given release_version already exists in index, please check index.json or use another version: ");
}

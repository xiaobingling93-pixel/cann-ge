/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"
#include "graph/operator.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph_metadef/register/register.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "base/err_mgr.h"

#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdlib.h>

#include "framework/common/debug/ge_log.h"
#include "register/op_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "register/op_compile_info_base.h"
#include "op_tiling.h"
#include "register/op_check_register.h"
#include "register/op_check.h"
#include "register/tilingdata_base.h"

#include "graph/graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/attr_value.h"

#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"

#include "proto/tensorflow/attr_value.pb.h"
#include "proto/tensorflow/node_def.pb.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "common/util/tiling_utils.h"
#include "hcom/hcom_topo_info.h"
#include <op_desc_utils_ex.h>
#include "faker/space_registry_faker.h"

using namespace domi;
using namespace ge;
using namespace optiling;
namespace ge {
void to_json(nlohmann::json &j, const HcomTopoInfo::TopoLevelDesc &desc);
void from_json(const nlohmann::json &j, HcomTopoInfo::TopoLevelDesc &desc);
void to_json(nlohmann::json &j, const HcomTopoInfo::TopoInfo &info);
void from_json(const nlohmann::json &j, HcomTopoInfo::TopoInfo &info);
}
namespace {
REG_OP(AddUt)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(AddUt);
// infer from output
REG_OP(FixIOOp_OutputIsFix)
    .INPUT(fix_input1, "T")
        .INPUT(fix_input2, "T")
        .OUTPUT(fix_output, "T2")
        .DATATYPE(T2, TensorType({DT_BOOL}))
        .OP_END_FACTORY_REG(FixIOOp_OutputIsFix);
}
class CompileInfoJson : public CompileInfoBase {
 public:
  CompileInfoJson(const std::string &json) : json_str_(json) {}
  ~CompileInfoJson() {}

 private:
  std::string json_str_;
};

namespace {
struct StubCompileInfo : public CompileInfoBase {
  int64_t stub_ = 2;
};

void *CreateCompileInfo() {
  return new StubCompileInfo();
}

void DeleteCompileInfo(void *compile_info) {
  delete reinterpret_cast<StubCompileInfo *>(compile_info);
}

UINT32 OpTilingStubNew(gert::TilingContext *kernel_context) {
  auto tensor_without_data = kernel_context->GetInputTensor(1);
  EXPECT_EQ(tensor_without_data->GetAddr(), nullptr);
  EXPECT_EQ(tensor_without_data->GetStorageShape(), gert::Shape({5, 5, 5, 5}));
  EXPECT_EQ(tensor_without_data->GetOriginShape(), gert::Shape({5, 5, 5, 5}));
  auto tensor = kernel_context->GetInputTensor(0);
  EXPECT_EQ(tensor->GetShape().GetStorageShape().GetDimNum(), 4);
  gert::Shape expect_shape({4, 4, 4, 4});
  EXPECT_EQ(tensor->GetShape().GetStorageShape(), expect_shape);
  EXPECT_EQ(tensor->GetDataType(), DT_INT8);
  EXPECT_EQ((tensor->GetData<int8_t>())[3], 4);
  EXPECT_EQ((tensor->GetData<int8_t>())[2], 3);
  EXPECT_EQ((tensor->GetData<int8_t>())[1], 2);
  EXPECT_EQ((tensor->GetData<int8_t>())[0], 1);
  EXPECT_EQ(tensor->GetFormat().GetStorageFormat(), FORMAT_ND);
  gert::Shape expect_shape2({9, 9, 9, 9});
  EXPECT_TRUE(kernel_context->GetOutputShape(0)->GetStorageShape() == expect_shape2);
  auto shape = kernel_context->GetInputShape(1);
  EXPECT_TRUE(*shape == gert::StorageShape({5, 5, 5, 5}, {5, 5, 5, 5}));
  auto ci = kernel_context->GetCompileInfo();
  EXPECT_EQ(reinterpret_cast<const StubCompileInfo *>(ci)->stub_, 1);

  EXPECT_EQ(kernel_context->GetAttrs()->GetAttrNum(), 5);
  std::vector<int64_t> expect_attr = {1, 2, 3, 4};
  for (size_t i = 0UL; i < 4UL; ++i) {
    EXPECT_EQ(reinterpret_cast<const int64_t *>(
                  kernel_context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0)->GetData())[i],
              expect_attr[i]);
  }
  EXPECT_EQ(*kernel_context->GetAttrs()->GetAttrPointer<int8_t>(1), 99);

  std::vector<int64_t> expect_attr3 = {2147483647, 2147483648, 9223372036854775807};
  for (size_t i = 0UL; i < 2UL; ++i) {
    EXPECT_EQ(reinterpret_cast<const int64_t *>(
                  kernel_context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(3)->GetData())[i],
              expect_attr3[i]);
  }
  kernel_context->SetBlockDim(2);
  kernel_context->SetAicpuBlockDim(4);
  kernel_context->SetNeedAtomic(true);
  kernel_context->SetTilingKey(78);
  *kernel_context->GetWorkspaceSizes(1) = 12;
  kernel_context->GetRawTilingData()->Append<uint8_t>(6);
  kernel_context->GetRawTilingData()->Append<uint8_t>(7);
  kernel_context->GetRawTilingData()->Append<uint8_t>(8);
  kernel_context->GetRawTilingData()->Append<uint8_t>(9);
  kernel_context->GetRawTilingData()->Append<uint8_t>(10);
  return ge::GRAPH_SUCCESS;
}

UINT32 OpTilingParseStubNew(gert::KernelContext *kernel_context) {
  auto ci = kernel_context->GetOutputPointer<StubCompileInfo>(0);
  ci->stub_ = 1;
  return ge::GRAPH_SUCCESS;
}

UINT32 OpTilingStubNewWithName(gert::TilingContext *kernel_context) {
  std::string node_name = kernel_context->GetNodeName();
  EXPECT_EQ(node_name, "test");
  return ge::GRAPH_SUCCESS;
}

UINT32 OpTilingStubV5(gert::TilingContext *kernel_context) {
  auto tensor = kernel_context->GetInputTensor(0);
  std::vector<float> real_data = {1.1, 2.1, 3.1, 4.1};
  for (size_t i = 0UL; i < 4UL; ++i) {
    EXPECT_EQ((tensor->GetData<uint16_t>())[i], optiling::Float32ToFloat16(real_data[i]));
  }
  return ge::GRAPH_SUCCESS;
}

UINT32 OpTilingStubBf16(gert::TilingContext *kernel_context) {
  auto tensor = kernel_context->GetInputTensor(0);
  std::vector<float> real_data = {1.1, 2.1, 3.1, 4.1};
  for (size_t i = 0UL; i < 4UL; ++i) {
    EXPECT_EQ((tensor->GetData<uint16_t>())[i], optiling::Float32ToBfloat16(real_data[i]));
  }
  (void)kernel_context->SetNeedAtomic(true);
  return ge::GRAPH_SUCCESS;
}

UINT32 OpTilingStubNewWithDynamicInput(gert::TilingContext *kernel_context) {
  auto shape = kernel_context->GetDynamicInputShape(0, 0);
  EXPECT_EQ(*shape, gert::StorageShape( {4, 256, 200, 336}, {4, 16, 200, 336, 16}));
  auto shape0_1 = kernel_context->GetDynamicInputShape(0, 1);
  EXPECT_EQ(*shape0_1, gert::StorageShape( {4, 256, 100, 168}, {4, 16, 100, 168, 16}));
  auto shape_optional_1 = kernel_context->GetOptionalInputShape(1);
  EXPECT_EQ(shape_optional_1, nullptr);
  auto shape_optional_2 = kernel_context->GetOptionalInputShape(2);
  EXPECT_EQ(shape_optional_2, nullptr);
  auto shape_2 = kernel_context->GetDynamicInputShape(3, 0);
  EXPECT_EQ(*shape_2, gert::StorageShape({100, 5}, {100, 5}));
  auto input_1 = kernel_context->GetInputShape(4);
  EXPECT_EQ(*input_1, gert::StorageShape({100, 5}, {100, 5}));
  auto output = kernel_context->GetOutputShape(0);
  EXPECT_EQ(*output, gert::StorageShape({9, 9, 9, 9}, {9, 9, 9, 9}));
  EXPECT_EQ(kernel_context->GetComputeNodeInputNum(), 5);
  EXPECT_EQ(kernel_context->GetComputeNodeOutputNum(), 1);
  return ge::GRAPH_SUCCESS;
}

UINT32 DefaultOptilingStub(gert::TilingContext *kernel_context) {
  (void)kernel_context->SetNeedAtomic(true);
  return ge::GRAPH_SUCCESS;
}

UINT32 OpTilingParseStubV5(gert::KernelContext *kernel_context) {
  auto av = kernel_context->GetOutput(0);
  av->Set(CreateCompileInfo(), DeleteCompileInfo);
  return ge::GRAPH_SUCCESS;
}

UINT32 OpTilingStubV6(gert::TilingContext *kernel_context) {
  auto input_desc = kernel_context->GetComputeNodeInfo()->GetInputTdInfo(0);
  EXPECT_EQ(input_desc->GetFormat().GetStorageFormat(),
            ge::GetFormatFromSub(static_cast<int32_t>(Format::FORMAT_FRACTAL_Z), 32));
  auto tensor = kernel_context->GetInputTensor(0);
  std::vector<float> real_data = {1.1, 2.1, 3.1, 4.1};
  for (size_t i = 0UL; i < 4UL; ++i) {
    EXPECT_EQ((tensor->GetData<uint16_t>())[i], optiling::Float32ToFloat16(real_data[i]));
  }
  //  强一致性计算紧急需求上库，ge暂时不能依赖metadef，已于BBIT及本地验证DT通过，后续补上
  //  auto deterministic_level = kernel_context->GetDeterministicLevel();
  //  EXPECT_EQ(deterministic_level, 0);
  return ge::GRAPH_SUCCESS;
}

bool op_tiling_stub_failed(const Operator &op, const utils::OpCompileInfo &compile_info, utils::OpRunInfo &run_info) {
  EXPECT_EQ(true, false);
  return true;
}

bool CheckErrorRetFormat(const std::string &ret_json_str) {
  EXPECT_NE(ret_json_str, "");
  nlohmann::json ret_json;
  try {
    ret_json = nlohmann::json::parse(ret_json_str);
    EXPECT_TRUE(ret_json.contains("ret_code"));
    EXPECT_EQ(ret_json.at("ret_code"), 1);
    EXPECT_TRUE(ret_json.contains("error_messages"));
    EXPECT_EQ(ret_json.at("error_messages").size(), 2);
    EXPECT_EQ(ret_json.at("error_messages")[0].at("type"), 2);
    EXPECT_EQ(ret_json.at("error_messages")[0].at("errorcode"), "E19999");
    EXPECT_EQ(0, std::string(ret_json.at("error_messages")[0].at("errormsg")).find("Parse attrs failed.[FUNC:ParseJson][FILE:op_tiling_py.cc]"));
    EXPECT_EQ(ret_json.at("error_messages")[1].at("type"), 2);
    EXPECT_EQ(ret_json.at("error_messages")[1].at("errorcode"), "E19999");
    EXPECT_EQ(0, std::string(ret_json.at("error_messages")[1].at("errormsg")).find("TbeOpTilingPyInterfaceOld failed.[FUNC:DoOpTilingForCompile][FILE:op_tiling_py.cc]"));
  } catch (const nlohmann::json::exception &e) {
    std::cout << "parse failed, reason: " << e.what() << std::endl;
    return false;
  }
  return true;
}

void ReInitErrorManager() {
  (void)error_message::GetErrMgrErrorMessage();
}
}  // namespace

class UtestRegister : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
    (void)error_message::GetErrMgrErrorMessage();
  }
};

extern "C" int TbeOpTilingPyInterfaceEx2(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse);

extern "C" int TbeOpTilingPyInterface(const char *optype, const char *compile_info, const char *compile_info_hash,
                                      const char *inputs, const char *outputs, const char *attrs, char *run_info_json,
                                      size_t run_info_len, uint64_t *elapse);

extern "C" const char *DoOpTilingForCompile(const char *optype, const char *compile_info, const char *compile_info_hash,
                                            const char *inputs,
                                            const char *outputs,
                                            const char *attrs,
                                            char *run_info_json,
                                            size_t run_info_len,
                                            uint64_t *elapse,
                                            const char *extra_info);

extern "C" Status TbeLoadSoAndSaveToRegistry(const char *so_path);

bool op_tiling_stub_v2(const Operator &op, const utils::OpCompileInfo &compile_info, utils::OpRunInfo &run_info) {
  return true;
}

bool op_tiling_stub_v3(const Operator &op, const void *value, OpRunInfoV2 &run_info) {
  return true;
}

void *op_parse_stub_v3(const Operator &op, const ge::AscendString &compile_info_json) {
  //  static void *p = new int(3);
  static int x = 1024;
  void *p = &x;
  return p;
}

bool op_tiling_stub_v4(const Operator &op, const CompileInfoPtr value, OpRunInfoV2 &run_info) {
  return true;
}

CompileInfoPtr op_parse_stub_v4(const Operator &op, const ge::AscendString &compile_info_json) {
  //  static void *p = new int(3);
  CompileInfoPtr info = std::make_shared<CompileInfoJson>("qwer");
  return info;
}


UINT32 OpTilingStubNewWithNullDesc(gert::TilingContext *kernel_context) {
  auto tensor_without_data = kernel_context->GetInputTensor(1);
  EXPECT_EQ(tensor_without_data->GetAddr(), nullptr);
  EXPECT_EQ(tensor_without_data->GetStorageShape(), gert::Shape({5, 5, 5, 5}));
  EXPECT_EQ(tensor_without_data->GetOriginShape(), gert::Shape({5, 5, 5, 5}));
  auto tensor = kernel_context->GetInputTensor(0);
  EXPECT_EQ(tensor->GetShape().GetStorageShape().GetDimNum(), 4);
  gert::Shape expect_shape({4, 4, 4, 4});
  EXPECT_EQ(tensor->GetShape().GetStorageShape(), expect_shape);
  EXPECT_EQ(tensor->GetDataType(), DT_FLOAT);
  EXPECT_EQ((tensor->GetData<float>())[3], -std::numeric_limits<float>::infinity());
  EXPECT_EQ(std::isnan((tensor->GetData<float>())[2]), true);
  EXPECT_EQ((tensor->GetData<float>())[1], 2.0);
  EXPECT_EQ((tensor->GetData<float>())[0], std::numeric_limits<float>::infinity());
  EXPECT_EQ(tensor->GetFormat().GetStorageFormat(), FORMAT_ND);
  gert::Shape expect_shape2({9, 9, 9, 9});
  EXPECT_TRUE(kernel_context->GetOutputShape(0)->GetStorageShape() == expect_shape2);
  auto shape = kernel_context->GetInputShape(1);
  EXPECT_TRUE(*shape == gert::StorageShape({5, 5, 5, 5}, {5, 5, 5, 5}));
  auto ci = kernel_context->GetCompileInfo();
  EXPECT_EQ(reinterpret_cast<const StubCompileInfo *>(ci)->stub_, 1);

  EXPECT_EQ(kernel_context->GetAttrs()->GetAttrNum(), 4);
  std::vector<float> expect_attr = {std::numeric_limits<float>::infinity(), 2.0,
                                    std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity()};
  for (size_t i = 0UL; i < 4UL; ++i) {
    if (i == 2U) {
      EXPECT_EQ(std::isnan(reinterpret_cast<const float *>(
                               kernel_context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0)->GetData())[i]), true);
      continue;
    }
    EXPECT_EQ(reinterpret_cast<const float *>(
                  kernel_context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0)->GetData())[i],
              expect_attr[i]);
  }
  EXPECT_EQ(*kernel_context->GetAttrs()->GetAttrPointer<float>(1), std::numeric_limits<float>::infinity());
  kernel_context->SetBlockDim(2);
  kernel_context->SetAicpuBlockDim(4);
  kernel_context->SetNeedAtomic(true);
  kernel_context->SetTilingKey(78);
  *kernel_context->GetWorkspaceSizes(1) = 12;
  kernel_context->GetRawTilingData()->Append<uint8_t>(6);
  kernel_context->GetRawTilingData()->Append<uint8_t>(7);
  kernel_context->GetRawTilingData()->Append<uint8_t>(8);
  kernel_context->GetRawTilingData()->Append<uint8_t>(9);
  kernel_context->GetRawTilingData()->Append<uint8_t>(10);
  return ge::GRAPH_SUCCESS;
}

void SupportInfNanWithNullDescInvalidTestCase(const nlohmann::json &input, const nlohmann::json &output,
                                              const nlohmann::json &attrs) {
  std::string input_str = input.dump();
  std::string output_str = output.dump();
  std::string attrs_str = attrs.dump();
  const char *op_type = "TestReluV2";
  const char *cmp_info = "";
  std::string runinfo(100, 'a');
  size_t size = 100;
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubNewWithNullDesc;
  op_impl_func->tiling_parse = OpTilingParseStubNew;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(runinfo.c_str()), size, elapse),
            0);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

REGISTER_OP_TILING_V2(ReluV2, op_tiling_stub_v2);
REGISTER_OP_TILING_V3(ReluV3, op_tiling_stub_v3, op_parse_stub_v3);
REGISTER_OP_TILING_V4(ReluV4, op_tiling_stub_v4, op_parse_stub_v4);

TEST_F(UtestRegister, test_register_dynamic_outputs_op_only_has_partial_output) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node_src = builder.AddNode("ParseSingleExample", "ParseSingleExample",
                                  {"serialized", "dense_defaults_0", "dense_defaults_1", "dense_defaults_2"},
                                  {"dense_values_0", "dense_values_1", "dense_values_2"});
  // build op_src attrs
  vector<string> dense_keys = {"image/class/lable", "image/encode", "image/format"};
  vector<DataType> t_dense = {DT_INT64, DT_STRING, DT_STRING};
  AttrUtils::SetListStr(node_src->GetOpDesc(), "dense_keys", dense_keys);
  AttrUtils::SetListStr(node_src->GetOpDesc(), "dense_shapes", {});
  AttrUtils::SetInt(node_src->GetOpDesc(), "num_sparse", 0);
  AttrUtils::SetListStr(node_src->GetOpDesc(), "sparse_keys", {});
  AttrUtils::SetListStr(node_src->GetOpDesc(), "sparse_types", {});
  AttrUtils::SetListDataType(node_src->GetOpDesc(), "Tdense", t_dense);
  auto graph = builder.GetGraph();

  // get op_src
  ge::Operator op_src = OpDescUtils::CreateOperatorFromNode(node_src);
  ge::Operator op_dst = ge::Operator("ParseSingleExample");
  std::shared_ptr<ge::OpDesc> op_desc_dst = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  op_desc_dst->AddRegisterInputName("dense_defaults");
  op_desc_dst->AddRegisterOutputName("sparse_indices");
  op_desc_dst->AddRegisterOutputName("sparse_values");
  op_desc_dst->AddRegisterOutputName("sparse_shapes");
  op_desc_dst->AddRegisterOutputName("dense_values");

  // simulate parse_single_example plugin
  std::vector<DynamicInputOutputInfo> value;
  DynamicInputOutputInfo input(kInput, "dense_defaults", 14, "Tdense", 6);
  value.push_back(input);
  DynamicInputOutputInfo output(kOutput, "sparse_indices", 14, "num_sparse", 10);
  value.push_back(output);
  DynamicInputOutputInfo output1(kOutput, "sparse_values", 13, "sparse_types", 12);
  value.push_back(output1);
  DynamicInputOutputInfo output2(kOutput, "sparse_shapes", 13, "num_sparse", 10);
  value.push_back(output2);
  DynamicInputOutputInfo output3(kOutput, "dense_values", 12, "Tdense", 6);
  value.push_back(output3);
  DynamicInputOutputInfo invalidput(kInvalid, "Invalid", 7, "Invalid", 7);
  value.push_back(invalidput);

  // pre_check
  EXPECT_EQ(op_dst.GetOutputsSize(), 0);
  auto ret = AutoMappingByOpFnDynamic(op_src, op_dst, value);

  // check add 3 output to op_dst
  EXPECT_EQ(ret, domi::SUCCESS);
  EXPECT_EQ(op_dst.GetOutputsSize(), 3);

  // for AutoMappingByOpFnDynamic failed test
  ge::Operator op_src_fail(nullptr);
  ret = AutoMappingByOpFnDynamic(op_src_fail, op_dst, value);
  EXPECT_EQ(ret, domi::FAILED);

  std::vector<DynamicInputOutputInfo> value_fail;
  ret = AutoMappingByOpFnDynamic(op_src, op_dst, value_fail);
  DynamicInputOutputInfo input_fail(kInput, "", 0, "", 0);
  value_fail.push_back(input_fail);
  ret = AutoMappingByOpFnDynamic(op_src, op_dst, value_fail);
  EXPECT_EQ(ret, domi::FAILED);
}

void GraphInit(domi::tensorflow::GraphDef &graph_def) {
  // add node, set info
  domi::tensorflow::NodeDef *placeholder0 = graph_def.add_node();
  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");

  // add node, set info, add edges
  domi::tensorflow::NodeDef *add0 = graph_def.add_node();
  add0->set_name("add0");
  add0->set_op("Add");
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  // 1. add node
  auto placeholder1 = graph_def.add_node();
  auto add1 = graph_def.add_node();
  auto mul0 = graph_def.add_node();
  auto mul1 = graph_def.add_node();
  auto add2 = graph_def.add_node();
  auto retval0 = graph_def.add_node();
  auto retval1 = graph_def.add_node();

  // 2. set info
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");
  add1->set_name("add1");
  add1->set_op("Add");
  add2->set_name("add2");
  add2->set_op("Add");
  mul0->set_name("mul0");
  mul0->set_op("Mul");
  mul1->set_name("mul1");
  mul1->set_op("Mul");
  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");

  // 3. add edges
  mul0->add_input("placeholder0");
  mul0->add_input("placeholder1");
  mul1->add_input("placeholder0");
  mul1->add_input("add0");
  mul1->add_input("^mul0");
  add1->add_input("mul0");
  add1->add_input("placeholder1");
  add2->add_input("mul1");
  add2->add_input("mul0");
  retval0->add_input("add2:0");
  retval1->add_input("add1:0");
}

int32_t AutoMappingSubgraphIndexInput(int32_t data_index) {
  return 0;
}
int32_t AutoMappingSubgraphIndexOutput(int32_t netoutput_index) {
  return 0;
}
Status AutoMappingSubgraphIndexInput2(int32_t data_index, int32_t &parent_input_index) {
  return domi::SUCCESS;
}
Status AutoMappingSubgraphIndexOutput2(int32_t netoutput_index, int32_t &parent_output_index) {
  parent_output_index++;
  return domi::SUCCESS;
}
Status AutoMappingSubgraphIndexOutput2Failed(int32_t netoutput_index, int32_t &parent_output_index) {
  return domi::FAILED;
}

TEST_F(UtestRegister, AutoMappingSubgraphIndex) {
  Status stat;
  auto builder = ut::GraphBuilder("root");
  auto output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto input = builder.AddNode("data", DATA, 1, 1);
  input->impl_->op_->impl_->meta_data_.type_ = "Data";
  auto func_node = builder.AddNode("func_node", FRAMEWORKOP, 1, 1);
  func_node->impl_->op_->impl_->meta_data_.type_ = "FrameworkOp";
  builder.AddDataEdge(input, 0, func_node, 0);
  builder.AddDataEdge(func_node, 0, output, 0);

  auto computeGraph = builder.GetGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(computeGraph);
  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput, AutoMappingSubgraphIndexOutput);
  EXPECT_EQ(stat, domi::FAILED);
}

TEST_F(UtestRegister, AutoMappingSubgraphIndexByDataNode) {
  Status stat;
  auto builder = ut::GraphBuilder("root");
  auto output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto func_node = builder.AddNode("func_node", PARTITIONEDCALL, 1, 1);
  builder.AddDataEdge(func_node, 0, output, 0);

  auto computeGraph = builder.GetGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(computeGraph);
  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput2, AutoMappingSubgraphIndexOutput2);
  EXPECT_EQ(stat, domi::SUCCESS);

  auto input = builder.AddNode("Retval", DATA, 1, 1);
  input->impl_->op_->impl_->meta_data_.type_ = "_Retval";
  AttrUtils::SetInt(input->GetOpDesc(), "retval_index", 0);
  builder.AddDataEdge(input, 0, func_node, 0);
  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput2, AutoMappingSubgraphIndexOutput2);
  EXPECT_EQ(stat, domi::SUCCESS);
}

TEST_F(UtestRegister, AutoMappingSubgraphIndexByDataNode2) {
  Status stat;
  auto builder = ut::GraphBuilder("root");
  auto input = builder.AddNode("index", DATA, 1, 1);
  input->impl_->op_->impl_->meta_data_.type_ = "Data";
  AttrUtils::SetInt(input->GetOpDesc(), "index", 0);
  auto output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto func_node = builder.AddNode("func_node", PARTITIONEDCALL, 1, 1);
  builder.AddDataEdge(input, 0, func_node, 0);
  builder.AddDataEdge(func_node, 0, output, 0);

  auto computeGraph = builder.GetGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(computeGraph);
  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput2, AutoMappingSubgraphIndexOutput2);
  EXPECT_EQ(stat, domi::SUCCESS);
}

TEST_F(UtestRegister, AutoMappingSubgraphOutputFail) {
  Status stat;
  auto builder = ut::GraphBuilder("root");
  auto output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto input = builder.AddNode("data", DATA, 1, 1);
  input->impl_->op_->impl_->meta_data_.type_ = "Data";
  auto func_node = builder.AddNode("func_node", FRAMEWORKOP, 1, 1);
  func_node->impl_->op_->impl_->meta_data_.type_ = "FrameworkOp";
  builder.AddDataEdge(input, 0, func_node, 0);
  builder.AddDataEdge(func_node, 0, output, 0);

  auto computeGraph = builder.GetGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(computeGraph);

  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput2, AutoMappingSubgraphIndexOutput2Failed);
  EXPECT_EQ(stat, domi::FAILED);
}

TEST_F(UtestRegister, AutoMappingFnDynamicInputTest) {
  Status stat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);
  map<std::string, std::pair<std::string, std::string>> name_attr_value;
  name_attr_value.insert(make_pair(std::string("in"), make_pair(std::string("dynamicName1"), std::string("int"))));
  name_attr_value.insert(make_pair(std::string("out"), make_pair(std::string("dynamicName2"), std::string("float"))));

  ge::Operator op_dst = ge::Operator("Add", "int");
  const domi::tensorflow::NodeDef *node;

  int32_t node_size = graph_def.node_size();
  for (int i = 0; i < node_size; i++) {
    node = graph_def.mutable_node(i);
    stat = AutoMappingFnDynamic(node, op_dst, name_attr_value, 1, 1);
    EXPECT_EQ(stat, domi::SUCCESS);
  }
}

TEST_F(UtestRegister, AutoMappingFnDynamicInput) {
  Status retStat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);

  ge::Operator op_dst = ge::Operator("Add", "int");
  domi::tensorflow::NodeDef *node = graph_def.mutable_node(0);

  // test for add attrs
  map<std::string, std::pair<std::string, std::string>> name_attrs;
  domi::tensorflow::AttrValue inValue;
  inValue.set_s(std::string("stringValue"));
  inValue.set_i(66);
  node->mutable_attr()->insert({"inVal", inValue});
  name_attrs.insert(make_pair(std::string("in"), make_pair(std::string("inName1"), std::string("inVal"))));
  retStat = AutoMappingFnDynamic(node, op_dst, name_attrs, 1, 1);
  EXPECT_EQ(retStat, domi::SUCCESS);
}

TEST_F(UtestRegister, AutoMappingFnDynamicOutput) {
  Status retStat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);

  ge::Operator op_dst = ge::Operator("Add", "int");
  domi::tensorflow::NodeDef *node = graph_def.mutable_node(0);

  // test for add attrs
  map<std::string, std::pair<std::string, std::string>> name_attrs;
  domi::tensorflow::AttrValue outValue;
  outValue.set_b(true);
  outValue.set_i(88);
  node->mutable_attr()->insert({"outVal", outValue});
  name_attrs.insert(make_pair(std::string("out"), make_pair(std::string("outName1"), std::string("outVal"))));
  retStat = AutoMappingFnDynamic(node, op_dst, name_attrs, 1, 1);
  EXPECT_EQ(retStat, domi::SUCCESS);
}

TEST_F(UtestRegister, AutoMappingFunctionkFunc) {
  Status retStat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);

  ge::Operator op_dst = ge::Operator("Add", "int");
  op_dst.SubgraphRegister("subVal", true);
  op_dst.SubgraphCountRegister("subVal", 6);

  // test for add attrs
  domi::tensorflow::NodeDef *node = graph_def.mutable_node(0);
  map<std::string, std::pair<std::string, std::string>> name_attrs;
  domi::tensorflow::AttrValue attrValue;
  attrValue.set_i(88);
  domi::tensorflow::NameAttrList *nameAttrList = new domi::tensorflow::NameAttrList();
  nameAttrList->set_name("nameAttrList");
  attrValue.unsafe_arena_set_allocated_func(nameAttrList);
  node->mutable_attr()->insert({"subVal", attrValue});
  name_attrs.insert(make_pair(std::string("out"), make_pair(std::string("outName1"), std::string("subVal"))));
  retStat = AutoMappingFnDynamic(node, op_dst, name_attrs, 1, 1);
  EXPECT_EQ(retStat, domi::FAILED);
}

TEST_F(UtestRegister, AutoMappingFunctionkList) {
  Status retStat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);

  ge::Operator op_dst = ge::Operator("Add", "int");
  op_dst.SubgraphRegister("subVal", true);

  // test for add attrs
  domi::tensorflow::NodeDef *node = graph_def.mutable_node(0);
  map<std::string, std::pair<std::string, std::string>> name_attrs;
  domi::tensorflow::AttrValue attrValue;
  attrValue.set_i(88);
  domi::tensorflow::AttrValue_ListValue *attrValListVal = new domi::tensorflow::AttrValue_ListValue();
  attrValListVal->add_s("list0");
  attrValListVal->add_s("list1");
  attrValue.unsafe_arena_set_allocated_list(attrValListVal);
  // list.func
  domi::tensorflow::NameAttrList *nameAttrList = new domi::tensorflow::NameAttrList();
  nameAttrList->set_name("nameAttrList");
  attrValListVal->add_func();

  node->mutable_attr()->insert({"subVal", attrValue});
  name_attrs.insert(make_pair(std::string("out"), make_pair(std::string("outName1"), std::string("subVal"))));
  retStat = AutoMappingFnDynamic(node, op_dst, name_attrs, 1, 1);
  EXPECT_EQ(retStat, domi::SUCCESS);
  delete nameAttrList;
}

domi::Status inputFunc(int32_t data_index, int32_t &parent_input_index) {
  parent_input_index++;
  return (parent_input_index < 0) ? domi::FAILED : domi::SUCCESS;
}

domi::Status outputFunc(int32_t netoutput_index, int32_t &parent_output_index) {
  parent_output_index++;
  return (parent_output_index < 2) ? domi::FAILED : domi::SUCCESS;
}

domi::Status AutoMappingSubgraphIOIndexFuncCB(
    const ge::Graph &graph, const std::function<Status(int32_t data_index, int32_t &parent_input_index)> &input,
    const std::function<Status(int32_t netoutput_index, int32_t &parent_output_index)> &output) {
  static int test_idx = -2;

  switch (test_idx) {
    case -2:
      return input(0, test_idx);
    case -1:
      return input(0, test_idx);
    case 0:
      return output(0, test_idx);
    case 1:
      return output(0, test_idx);
  }
  return domi::SUCCESS;
}

TEST_F(UtestRegister, FrameworkRegistryTest) {
  auto TENSORFLOW = domi::TENSORFLOW;
  REGISTER_AUTOMAPPING_SUBGRAPH_IO_INDEX_FUNC(TENSORFLOW, AutoMappingSubgraphIOIndexFuncCB);

  FrameworkRegistry &cur = FrameworkRegistry::Instance();
  cur.AddAutoMappingSubgraphIOIndexFunc(domi::CAFFE, AutoMappingSubgraphIOIndexFuncCB);

  const ge::Graph graph("graph_test");
  AutoMappingSubgraphIOIndexFunc func = cur.GetAutoMappingSubgraphIOIndexFunc(domi::CAFFE);
  EXPECT_EQ(func(graph, inputFunc, outputFunc), domi::FAILED);
  EXPECT_EQ(func(graph, inputFunc, outputFunc), domi::SUCCESS);
  EXPECT_EQ(func(graph, inputFunc, outputFunc), domi::FAILED);
  EXPECT_EQ(func(graph, inputFunc, outputFunc), domi::SUCCESS);
}

TEST_F(UtestRegister, OpRegistrationDataWithNoImpl) {
  OpRegistrationData opRegData(std::string("OmOptype"));
  opRegData.impl_.reset();

  EXPECT_EQ(opRegData.GetOmOptype() == "", true);
  EXPECT_EQ(opRegData.GetFrameworkType(), domi::FRAMEWORK_RESERVED);
  EXPECT_EQ(opRegData.GetOriginOpTypeSet().empty(), true);
  EXPECT_EQ(opRegData.GetParseParamFn(), nullptr);
  EXPECT_EQ(opRegData.GetParseParamByOperatorFn(), nullptr);
  EXPECT_EQ(opRegData.GetFusionParseParamFn(), nullptr);
  EXPECT_EQ(opRegData.GetFusionParseParamByOpFn(), nullptr);
  EXPECT_EQ(opRegData.GetImplyType(), domi::ImplyType::BUILDIN);
  EXPECT_EQ(opRegData.GetParseSubgraphPostFn(), nullptr);
  EXPECT_EQ(opRegData.GetParseOpToGraphFn(), nullptr);
  ParseSubgraphFuncV2 func;
  EXPECT_EQ(opRegData.GetParseSubgraphPostFn(func), domi::FAILED);
}

TEST_F(UtestRegister, OmOptypeTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));
  OpReceiver oprcver(opRegData);
  opRegData.GetOmOptype();

  AscendString OmOptype;
  Status stat = opRegData.GetOmOptype(OmOptype);
  EXPECT_EQ(stat, domi::SUCCESS);
}

TEST_F(UtestRegister, FrameworkTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  opRegData.FrameworkType(domi::MINDSPORE);
  EXPECT_EQ(opRegData.GetFrameworkType(), domi::MINDSPORE);
}

TEST_F(UtestRegister, OriOpTypeTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));
  OpRegistrationData opRegData2("OmOptype2");

  std::initializer_list<std::string> OptypeList1{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList1);
  std::vector<AscendString> OptypeList2 = {AscendString("Div"), AscendString("Mul")};
  opRegData.OriginOpType(OptypeList2);

  opRegData2.OriginOpType(std::string("Add"));
  opRegData2.OriginOpType("Sub");

  opRegData.GetOriginOpTypeSet();
  std::set<ge::AscendString> opTypeSet;
  Status stat = opRegData.GetOriginOpTypeSet(opTypeSet);
  EXPECT_EQ(stat, domi::SUCCESS);
}

TEST_F(UtestRegister, OpRegistryImplyTypeTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);
  std::vector<AscendString> OptypeList2 = {AscendString("Div"), AscendString("Mul")};
  opRegData.OriginOpType(OptypeList2);

  // set ImplyType
  opRegData.ImplyType(domi::ImplyType::CUSTOM);
  EXPECT_EQ(opRegData.GetImplyType(), domi::ImplyType::CUSTOM);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  domi::ImplyType implType = opReg->GetImplyTypeByOriOpType(std::string("Add"));
  EXPECT_EQ(implType, domi::ImplyType::CUSTOM);

  implType = opReg->GetImplyType(std::string("OmOptype"));
  EXPECT_EQ(implType, domi::ImplyType::CUSTOM);
  implType = opReg->GetImplyType(std::string("strOmOptype"));
  EXPECT_EQ(implType, domi::ImplyType::BUILDIN);

  vector<std::string> vecOpType;
  vecOpType.clear();
  opReg->GetOpTypeByImplyType(vecOpType, domi::ImplyType::CUSTOM);
  EXPECT_EQ(vecOpType.empty(), false);
  vecOpType.clear();
  opReg->GetOpTypeByImplyType(vecOpType, domi::ImplyType::AI_CPU);
  EXPECT_EQ(vecOpType.empty(), true);
}

TEST_F(UtestRegister, DelInputWithTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));
  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.ParseParamsFn(domi::AutoMappingFn);
  EXPECT_NE(opRegData.GetParseParamFn(), nullptr);

  // insert input into vector
  const vector<int> input_order{0, 1, 3, 2};
  opRegData.InputReorderVector(input_order);

  opRegData.DelInputWithCond(1, std::string("attrName_1"), true);
  opRegData.DelInputWithCond(2, "attrName_2", false);

  opRegData.DelInputWithOriginalType(3, std::string("Add"));
  opRegData.DelInputWithOriginalType(4, "Sub");

  OpRegistry *opReg = OpRegistry::Instance();
  ASSERT_NE(opReg, nullptr);
  bool retBool = opReg->Register(opRegData);
  ASSERT_EQ(retBool, true);

  std::vector<RemoveInputConfigure> rmConfigVec;
  rmConfigVec = opReg->GetRemoveInputConfigure(std::string("Add"));
  EXPECT_EQ(rmConfigVec.empty(), true);
  rmConfigVec = opReg->GetRemoveInputConfigure(std::string("Mul"));
  EXPECT_EQ(rmConfigVec.empty(), true);
  rmConfigVec = opReg->GetRemoveInputConfigure(std::string("Mul666"));
  EXPECT_EQ(rmConfigVec.empty(), true);
}

TEST_F(UtestRegister, GetOmTypeByOriOpTypeTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);
  std::string om_type;
  EXPECT_EQ(opReg->GetOmTypeByOriOpType(std::string("Sub"), om_type), true);
  EXPECT_EQ(opReg->GetOmTypeByOriOpType(std::string("Sub1"), om_type), false);
}

domi::Status FusionParseParamsFnCB(const std::vector<const google::protobuf::Message *> Msg, ge::Operator &Op) {
  return domi::SUCCESS;
}
domi::Status FusionParseParamsFnCB2(const std::vector<ge::Operator> &VecOp, ge::Operator &Op) {
  return domi::FAILED;
}
domi::Status ParseSubgraphPostFnCB(const std::string &subgraph_name, const ge::Graph &graph) {
  return domi::SUCCESS;
}
domi::Status ParseSubgraphPostFnCB2(const ge::AscendString &subgraph_name, const ge::Graph &graph) {
  return domi::SUCCESS;
}
domi::Status ParseOpToGraphFnCB(const ge::Operator &Op, ge::Graph &Graph) {
  return domi::SUCCESS;
}

TEST_F(UtestRegister, ParseParamFuncTest) {
  const std::string strOmOptype = "OmOptype";
  OpRegistrationData opRegData(strOmOptype);

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);
  std::vector<AscendString> OptypeListAStr = {AscendString("Div"), AscendString("Mul")};
  opRegData.OriginOpType(OptypeListAStr);

  opRegData.ParseParamsFn(domi::AutoMappingFn);
  EXPECT_NE(opRegData.GetParseParamFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetParseParamFunc(std::string("OmOptype1"), std::string("Sub")), nullptr);
  EXPECT_EQ(opReg->GetParseParamFunc(std::string("OmOptype"), std::string("Sub")), nullptr);
}

TEST_F(UtestRegister, FusionParseParamFuncTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.FusionParseParamsFn(FusionParseParamsFnCB);
  EXPECT_NE(opRegData.GetFusionParseParamFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetFusionParseParamFunc(std::string("OmOptype"), std::string("Sub")), nullptr);
  EXPECT_EQ(opReg->GetFusionParseParamFunc(std::string("OmOptype1"), std::string("Sub")), nullptr);
}

TEST_F(UtestRegister, GetParseOpToGraphFuncTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.ParseOpToGraphFn(ParseOpToGraphFnCB);
  EXPECT_NE(opRegData.GetParseOpToGraphFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);
  std::string om_type;

  EXPECT_EQ(opReg->GetParseOpToGraphFunc(std::string("OmOptype"), std::string("Add")), nullptr);
  EXPECT_EQ(opReg->GetParseOpToGraphFunc(std::string("OmOptype"), std::string("Mul")), nullptr);
}

TEST_F(UtestRegister, ParseParamByOperatorFuncTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.ParseParamsByOperatorFn(domi::AutoMappingByOpFn);
  EXPECT_NE(opRegData.GetParseParamByOperatorFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetParseParamByOperatorFunc(std::string("int")), nullptr);
  EXPECT_EQ(opReg->GetParseParamByOperatorFunc(std::string("Add")), nullptr);
}

TEST_F(UtestRegister, FusionParseParamByOpFuncTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.FusionParseParamsFn(FusionParseParamsFnCB);
  EXPECT_NE(opRegData.GetFusionParseParamFn(), nullptr);

  opRegData.FusionParseParamsFn(FusionParseParamsFnCB2);
  EXPECT_NE(opRegData.GetFusionParseParamByOpFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetFusionParseParamByOpFunc(std::string("strOmOptype"), std::string("Add")), nullptr);
  EXPECT_EQ(opReg->GetFusionParseParamByOpFunc(std::string("OmOptype"), std::string("Add")), nullptr);
}

TEST_F(UtestRegister, ParseSubgraphPostFnTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.ParseSubgraphPostFn(ParseSubgraphPostFnCB);
  EXPECT_NE(opRegData.GetParseSubgraphPostFn(), nullptr);

  opRegData.ParseSubgraphPostFn(ParseSubgraphPostFnCB2);
  EXPECT_NE(opRegData.GetParseSubgraphPostFn(), nullptr);

  ParseSubgraphFuncV2 Getfunc;
  opRegData.GetParseSubgraphPostFn(Getfunc);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetParseSubgraphPostFunc(std::string("strOmOptype")), nullptr);
  EXPECT_EQ(opReg->GetParseSubgraphPostFunc(std::string("OmOptype")), nullptr);

  domi::ParseSubgraphFuncV2 parse_subgraph_func;
  EXPECT_EQ(opReg->GetParseSubgraphPostFunc(std::string("OmOptype"), parse_subgraph_func), domi::SUCCESS);
  EXPECT_EQ(opReg->GetParseSubgraphPostFunc(std::string("strOmOptype"), parse_subgraph_func), domi::FAILED);
}

TEST_F(UtestRegister, optiling_py_interface) {
  EXPECT_NO_THROW(
    const nlohmann::json j = R"([
        {
            "name": "test_0",
            "dtype": "int8",
            "value": 1,
            "const_value": [
              1,
              1,
              1,
              1
            ],
            "shape": [
              4,
              4,
              4,
              4
            ],
            "format": "ND"
        },
        {
            "name": "test_1",
            "dtype": "list_int",
            "value": [
              1,
              1,
              1,
              1
            ]
        },
        {
            "name": "test_2"
        },
        {
            "name": "test_2",
            "dtype": "list_list_int",
            "value": [
              [1, 2],
              [1, 2],
              [1, 2],
              [1, 2]
            ]
        },
        {
            "name": "test_0",
            "dtype": "list_list_int64",
            "value": [
              [1, 2],
              [1, 2],
              [1, 2],
              [1, 2]
            ]
        },
        {
            "name": "test_3",
            "dtype": "test",
            "value": "1"
        }
        ])"_json;

    std::string json_str = j.dump();
    ge::Operator op("NULL");
    const char *optype = "ReluV2";
    const char *optype_v3 = "ReluV3";
    const char *optype_v4 = "ReluV4";
    const char *cmp_info = "{\"_common_info\":[0,16,48,1,1,0,0],\"_is_ori_last_transpose\":0,\"_pattern\":\"Transdata\","
                           "\"_permute\":[0,2,1,3],\"_sgt_cube_vector_core_type\":\"VectorCore\",\"_src_fuse\":[0,1,3],"
                           "\"_src_pad_mode\":[0,0,2],\"_src_pad_var\":[1,1,16],\"_ub_info"
                           "\":[[48512,24192],[-1],[-1],[-1]],\"device_id\":\"0\"}";
    char *runinfo = const_cast<char *>("");
    size_t size = 3;
    const char *cmp_info_hash = "";
    uint64_t *elapse = nullptr;
    const char *attrs = json_str.c_str();
    TbeOpTilingPyInterface(optype, cmp_info, cmp_info_hash, attrs, attrs, attrs, runinfo, size, elapse);
    TbeOpTilingPyInterface(optype_v3, cmp_info, cmp_info_hash, attrs, attrs, attrs, runinfo, size, elapse);
    TbeOpTilingPyInterface(optype_v4, cmp_info, cmp_info_hash, attrs, attrs, attrs, runinfo, size, elapse);
    TbeOpTilingPyInterfaceEx2(optype, cmp_info, attrs, attrs, runinfo, size, cmp_info_hash, elapse);
    TbeOpTilingPyInterfaceEx2(optype_v3, cmp_info, attrs, attrs, runinfo, size, cmp_info_hash, elapse);
    TbeOpTilingPyInterfaceEx2(optype_v4, cmp_info, attrs, attrs, runinfo, size, cmp_info_hash, elapse);
  );
}

TEST_F(UtestRegister, new_optiling_py_interface_ok) {
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "int32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int32","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_int64","value": [1,2, 3, 4]},
{ "name": "attr_1","dtype": "int","value": 99},
{ "name": "attr_2","dtype": "list_int32","value": [1, 2, 3, 4]},
{ "name": "attr_3","dtype": "list_int","value": [2147483647, 2147483648, 9223372036854775807]},
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  const char *op_type = "TestReluV2";
  const char *cmp_info = "";
  std::string result =
      R"({"aicpu_block_dim":4,"block_dim":2,"clear_atomic":true,"local_memory_size":0,"schedule_mode":0,"tiling_cond":0,"tiling_data":"060708090A","tiling_key":78,"workspaces":[12]})";
  size_t size = result.length();
  std::string runinfo(size, 'a');
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubNew;
  op_impl_func->tiling_parse = OpTilingParseStubNew;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(runinfo.c_str()), size + 1U, elapse),
            1);
  EXPECT_EQ(result, runinfo);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_fail_with_invalid_const_value) {
  // int9999 is invalid data type
  const nlohmann::json input = R"([
  {"name": "test_0","dtype": "int9999", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"}])"_json;
  std::string input_str = input.dump();
  std::string output_str = " ";
  std::string attrs_str = " ";
  const char *op_type = "TestReluV2";
  const char *cmp_info = "";
  size_t size = 150;
  std::string runinfo(size, 'a');
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubNew;
  op_impl_func->tiling_parse = OpTilingParseStubNew;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(runinfo.c_str()), size, elapse),
            0);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_fail_with_invalid_attr) {
  std::string input_str = " ";
  std::string output_str = " ";
  // int999 is invalid dtype
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_int64","value": [1,2, 3, 4]},
{ "name": "attr_1","dtype": "int9999","value": 99}])"_json;
  std::string attrs_str = attrs.dump();
  const char *op_type = "TestReluV2";
  const char *cmp_info = "";
  size_t size = 150;
  std::string runinfo(size, 'a');
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubNew;
  op_impl_func->tiling_parse = OpTilingParseStubNew;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(runinfo.c_str()), size, elapse),
            0);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_fail_without_params) {
  EXPECT_EQ(TbeOpTilingPyInterface(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr), 0);
}

TEST_F(UtestRegister, new_optiling_py_interface_ok_with_float_data) {
  const nlohmann::json input = R"([
{"name": "t0", "dtype": "float16","const_value": [1.1,2.1,3.1,4.1] ,"shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "ND"},
{"dtype": "int8", "shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "ND"}
])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const char *op_type = "TestReluV2";
  const char *cmp_info = "";

  size_t size = 161;
  std::string runinfo(size, 'a');

  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50},
{ "name": "group", "dtype": "str", "value": "empty"}])"_json;

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubV5;
  op_impl_func->tiling_parse = OpTilingParseStubV5;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse),
            1);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_ok_with_sub_format) {
  const nlohmann::json input = R"([
{"name": "t0", "dtype": "float16","const_value": [1.1,2.1,3.1,4.1] ,"shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "FRACTAL_Z", "sub_format" :32},
{"dtype": "int8", "shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "ND"}
])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const char *op_type = "TestReluV2";
  const char *cmp_info = "";
  std::string runinfo(161, 'a');
  size_t size = 161;
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50}, { "name": "test_name", "dtype": "list_int", "value": [50, 51]}])"_json;
  const nlohmann::json extra_infos = R"([
{ "op_name": "matmul_all_reduce", "rank_size": 1}])"_json;

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubV6;
  op_impl_func->tiling_parse = OpTilingParseStubV5;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(std::string(DoOpTilingForCompile(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                               attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse,
                               extra_infos.dump().c_str())),
            "{\"ret_code\":0}");
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_ok_with_extra_info_case2) {
  const nlohmann::json input = R"([
{"name": "t0", "dtype": "float16","const_value": [1.1,2.1,3.1,4.1] ,"shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "FRACTAL_Z", "sub_format" :32},
{"dtype": "int8", "shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "ND"}
])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const char *op_type = "MatmulAllreduce";
  const char *cmp_info = "";
  std::string runinfo(161, 'a');
  size_t size = 161;
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50}, { "name": "group", "dtype": "str", "value": "g0"}])"_json;
  ge::HcomTopoInfo::TopoInfo original;
  original.rank_size = 64;
  original.notify_handle = reinterpret_cast<void*>(0x1234); // 指针不会被序列化

  // 初始化topo_level_descs
  original.topo_level_descs[static_cast<int>(ge::HcomTopoInfo::TopoLevel::L0)] = {8, 16};
  original.topo_level_descs[static_cast<int>(ge::HcomTopoInfo::TopoLevel::L1)] = {4, 32};

  // 序列化
  nlohmann::json j = original;
  nlohmann::json j_wrapped;
  j_wrapped["hcom_topo_info"] = j;
  std::string extra_infos = j_wrapped.dump(2);
  std::cout << "Serialized JSON:\n" << extra_infos << "\n\n";

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubV6;
  op_impl_func->tiling_parse = OpTilingParseStubV5;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(std::string(DoOpTilingForCompile(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                             attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse,
                                             extra_infos.c_str())),
            "{\"ret_code\":0}");
  int64_t rank_size_get = -1;
  EXPECT_EQ(ge::HcomTopoInfo::Instance().GetGroupRankSize("g0", rank_size_get), ge::GRAPH_SUCCESS);
  EXPECT_EQ(original.rank_size, rank_size_get);
  auto topo_desc = ge::HcomTopoInfo::Instance().GetGroupTopoDesc("g0");
  EXPECT_NE(topo_desc, nullptr);
  EXPECT_EQ(((*topo_desc)[0]).comm_sets, 8);
  EXPECT_EQ(((*topo_desc)[0]).rank_size, 16);
  EXPECT_EQ(((*topo_desc)[1]).comm_sets, 4);
  EXPECT_EQ(((*topo_desc)[1]).rank_size, 32);
  void *notify_handle = nullptr;
  EXPECT_EQ(ge::HcomTopoInfo::Instance().GetGroupNotifyHandle("g0", notify_handle), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(notify_handle == nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_ok_with_extra_info_case3) {
  const nlohmann::json input = R"([
    {"name": "t0", "dtype": "float16","const_value": [1.1,2.1,3.1,4.1] ,"shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "FRACTAL_Z", "sub_format" :32},
    {"dtype": "int8", "shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "ND"}
  ])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
    {"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const char *op_type = "MatmulAllreduce";
  const char *cmp_info = "";
  std::string runinfo(161, 'a');
  size_t size = 161;
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
    { "name": "op_para_size", "dtype": "int", "value": 50}, { "name": "group_ep", "dtype": "str", "value": "g1"}, { "name": "group_tp", "dtype": "str", "value": "g2"}])"_json;
  ge::HcomTopoInfo::TopoInfo original;
  original.notify_handle = reinterpret_cast<void*>(0x1234); // 指针不会被序列化
  original.local_window_size = 100;

  // 初始化topo_level_descs
  original.topo_level_descs[static_cast<int>(ge::HcomTopoInfo::TopoLevel::L0)] = {8, 16};
  original.topo_level_descs[static_cast<int>(ge::HcomTopoInfo::TopoLevel::L1)] = {4, 32};

  // 序列化
  nlohmann::json j = original;
  nlohmann::json j_wrapped;
  j_wrapped["group_ep"] = j;
  j_wrapped["group_tp"] = j;
  std::string extra_infos = j_wrapped.dump(2);
  std::cout << "Serialized JSON:\n" << extra_infos << "\n\n";

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubV6;
  op_impl_func->tiling_parse = OpTilingParseStubV5;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(std::string(DoOpTilingForCompile(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                             attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse,
                                             extra_infos.c_str())),
  "{\"ret_code\":0}");
  uint64_t local_window_size = -1;
  EXPECT_EQ(ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize("g1", local_window_size), ge::GRAPH_SUCCESS);
  EXPECT_EQ(original.local_window_size, 100);
  auto topo_desc = ge::HcomTopoInfo::Instance().GetGroupTopoDesc("g1");
  EXPECT_NE(topo_desc, nullptr);
  EXPECT_EQ(((*topo_desc)[0]).comm_sets, 8);
  EXPECT_EQ(((*topo_desc)[0]).rank_size, 16);
  EXPECT_EQ(((*topo_desc)[1]).comm_sets, 4);
  EXPECT_EQ(((*topo_desc)[1]).rank_size, 32);
  void *notify_handle = nullptr;
  EXPECT_EQ(ge::HcomTopoInfo::Instance().GetGroupNotifyHandle("g1", notify_handle), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(notify_handle == nullptr);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_ok_with_extra_info_case_invalid) {
  const nlohmann::json input = R"([
{"name": "t0", "dtype": "float16","const_value": [1.1,2.1,3.1,4.1] ,"shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "FRACTAL_Z", "sub_format" :32},
{"dtype": "int8", "shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "ND"}
])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const char *op_type = "MatmulAllreduce";
  const char *cmp_info = "";
  std::string runinfo(161, 'a');
  size_t size = 161;
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50}, { "name": "group", "dtype": "str", "value": "g3"}])"_json;
  const nlohmann::json invalid_extra_infos = R"([
{
  "hcom_topo_info": {
    "rank_size": 64,
    "topo_level_descs": [
      {
        "comm_sets": 8,
        "rank_size": 16
      },
      {
        "comm_sets": 4,
        "rank_size": 32
      },
      {
        "comm_sets": 4,
        "rank_size": 32
      }
    ]
  }
}])"_json;

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubV6;
  op_impl_func->tiling_parse = OpTilingParseStubV5;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(std::string(DoOpTilingForCompile(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                             attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse,
                                             invalid_extra_infos.dump().c_str())),
            "{\"ret_code\":0}");
  int64_t rank_size_get = -1;
  EXPECT_EQ(ge::HcomTopoInfo::Instance().GetGroupRankSize("g3", rank_size_get), ge::GRAPH_SUCCESS);
  EXPECT_EQ(rank_size_get, 64);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_ok_auto_tiling) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DefaultImpl");
  op_impl_func->tiling = DefaultOptilingStub;
  op_impl_func->tiling_parse = OpTilingParseStubV5;

  // expect rt1 tiling not to work
  REGISTER_OP_TILING_V2(AutoTiling, op_tiling_stub_failed);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8","shape": [4,4,4,4],"format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const char *op_type = "AutoTiling";
  const char *cmp_info = "";
  size_t size = 161;
  std::string runinfo(size, 'a');
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse),
            1);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, NewOptilingInterface_Ok_WithEmptyTensor) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DefaultImpl");
  op_impl_func->tiling = DefaultOptilingStub;
  op_impl_func->tiling_parse = OpTilingParseStubV5;

  // expect rt1 tiling not to work
  REGISTER_OP_TILING_V2(AutoTiling, op_tiling_stub_failed);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8","shape": [0],"format": "ND", "const value": ""}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const char *op_type = "AutoTiling";
  const char *cmp_info = "";
  size_t size = 161;
  std::string runinfo(size, 'a');
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse),
            1);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, NewOptilingInterface_Ok_WithNodeName) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DefaultImpl");
  op_impl_func->tiling = OpTilingStubNewWithName;
  op_impl_func->tiling_parse = OpTilingParseStubV5;

  // expect rt1 tiling not to work
  REGISTER_OP_TILING_V2(AutoTiling, op_tiling_stub_failed);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8","shape": [0],"format": "ND", "const value": ""}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const nlohmann::json extra_info = R"({"op_name": "test"})"_json;
  std::string extra_info_str = extra_info.dump();
  const char *op_type = "AutoTiling";
  const char *cmp_info = "";
  std::string runinfo(161, 'a');
  size_t size = 161;
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  EXPECT_EQ(std::string(DoOpTilingForCompile(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                             attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse,
                                             extra_info_str.c_str())),
            "{\"ret_code\":0}");

  {
    const nlohmann::json invalid_attrs = R"([
{ "name": "op_para_size", "value": 50}])"_json;
    EXPECT_NE(std::string::npos, std::string(DoOpTilingForCompile(op_type,
                                             cmp_info,
                                             cmp_info_hash,
                                             input_str.c_str(),
                                             output_str.c_str(),
                                             invalid_attrs.dump().c_str(),
                                             const_cast<char *>(runinfo.c_str()),
                                             size,
                                             elapse,
                                             extra_info_str.c_str())).find("\"ret_code\":1}"));
    EXPECT_NE(std::string::npos, std::string(DoOpTilingForCompile(nullptr,
                                             cmp_info,
                                             cmp_info_hash,
                                             input_str.c_str(),
                                             output_str.c_str(),
                                             invalid_attrs.dump().c_str(),
                                             const_cast<char *>(runinfo.c_str()),
                                             size,
                                             elapse,
                                             extra_info_str.c_str())).find("\"ret_code\":1}"));
    EXPECT_NE(std::string::npos, std::string(DoOpTilingForCompile("TestReluV2",
                                             cmp_info,
                                             cmp_info_hash,
                                             input_str.c_str(),
                                             output_str.c_str(),
                                             invalid_attrs.dump().c_str(),
                                             const_cast<char *>(runinfo.c_str()),
                                             size,
                                             elapse,
                                             extra_info_str.c_str())).find("\"ret_code\":1}"));
    std::string ret_when_error = DoOpTilingForCompile("TestReluV2",
                                                      cmp_info,
                                                      cmp_info_hash,
                                                      input_str.c_str(),
                                                      output_str.c_str(),
                                                      invalid_attrs.dump().c_str(),
                                                      const_cast<char *>(runinfo.c_str()),
                                                      size,
                                                      elapse,
                                                      extra_info_str.c_str());
    EXPECT_TRUE(CheckErrorRetFormat(ret_when_error));
    ReInitErrorManager();
  }
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, NewOptilingInterface_Ok_WithDynamicInput) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl("DefaultImpl");
  op_impl_func->tiling = OpTilingStubNewWithDynamicInput;
  op_impl_func->tiling_parse = OpTilingParseStubV5;

  const nlohmann::json input = R"([
  [{
    "shape": [4, 16, 200, 336, 16],
    "ori_shape": [4, 256, 200, 336],
    "format": "NC1HWC0",
        "sub_format": 0,
        "ori_format": "NCHW",
        "dtype": "float16",
        "addr_type": 0,
        "total_shape": [4, 16, 200, 336, 16],
    "slice_offset": [],
        "L1_addr_offset": 0,
        "L1_fusion_type": -1,
        "L1_workspace_size": -1,
        "valid_shape": [],
        "split_index": 0,
        "atomic_type": "",
        "input_c_values": 256,
        "range": [ [4, 4], [16, 16], [200, 200], [336, 336], [16, 16] ],
    "param_name": "feats",
        "name": "feats_gm_0"
  },
  {
    "shape": [4, 16, 100, 168, 16],
    "ori_shape": [4, 256, 100, 168],
    "format": "NC1HWC0",
        "sub_format": 0,
        "ori_format": "NCHW",
        "dtype": "float16",
        "addr_type": 0,
        "total_shape": [4, 16, 100, 168, 16],
    "slice_offset": [],
        "L1_addr_offset": 0,
        "L1_fusion_type": -1,
        "L1_workspace_size": -1,
        "valid_shape": [],
        "split_index": 0,
        "atomic_type": "",
        "input_c_values": 256,
        "range": [ [4, 4], [16, 16], [100, 100], [168, 168], [16, 16] ],
    "param_name": "feats",
        "name": "feats_gm_1"
  },
  {
    "shape": [4, 16, 50, 84, 16],
    "ori_shape": [4, 256, 50, 84],
    "format": "NC1HWC0",
        "sub_format": 0,
        "ori_format": "NCHW",
        "dtype": "float16",
        "addr_type": 0,
        "total_shape": [4, 16, 50, 84, 16],
    "slice_offset": [],
        "L1_addr_offset": 0,
        "L1_fusion_type": -1,
        "L1_workspace_size": -1,
        "valid_shape": [],
        "split_index": 0,
        "atomic_type": "",
        "input_c_values": 256,
        "range": [ [4, 4], [16, 16], [50, 50], [84, 84], [16, 16] ],
    "param_name": "feats"
  },
  {
    "shape": [4, 16, 25, 42, 16],
    "ori_shape": [4, 256, 25, 42],
    "format": "NC1HWC0",
        "sub_format": 0,
        "ori_format": "NCHW",
        "dtype": "float16",
        "addr_type": 0,
        "total_shape": [4, 16, 25, 42, 16],
    "slice_offset": [],
        "L1_addr_offset": 0,
        "L1_fusion_type": -1,
        "L1_workspace_size": -1,
        "valid_shape": [],
        "split_index": 0,
        "atomic_type": "",
        "input_c_values": 256,
        "range": [ [4, 4], [16, 16], [25, 25], [42, 42], [16, 16] ],
    "param_name": "feats"
  }
  ],
  null,
  null,
  {
    "shape": [100, 5],
    "ori_shape": [100, 5],
    "format": "NCHW",
        "sub_format": 0,
        "ori_format": "NCHW",
        "dtype": "float16",
        "addr_type": 0,
        "total_shape": [100, 5],
    "slice_offset": [],
        "L1_addr_offset": 0,
        "L1_fusion_type": -1,
        "L1_workspace_size": -1,
        "valid_shape": [],
        "split_index": 0,
        "atomic_type": "",
        "input_c_values": 5,
        "range": [ [100, 100], [5, 5] ],
    "param_name": "rois"
  }])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const nlohmann::json extra_info = R"({"op_name": "test", "rank_size": 1})"_json;
  std::string extra_info_str = extra_info.dump();
  const char *op_type = "AutoTiling";
  const char *cmp_info = "";
  std::string runinfo(161, 'a');
  size_t size = 161;
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  EXPECT_EQ(std::string(DoOpTilingForCompile(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                           attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse,
                                           extra_info_str.c_str())),
            "{\"ret_code\":0}");
}

extern "C" int AscendCPyInterfaceCheckOp(const char *check_type, const char *optype, const char *inputs,
                                      const char *outputs, const char *attrs, char *result_info,
                                      size_t result_info_len);

extern "C" int AscendCPyInterfaceGeneralized(const char *optype, const char *inputs, const char *outputs,
                                          const char *attrs, const char *generalize_config, char *result_info,
                                          size_t result_info_len);

extern "C" int AscendCPyInterfaceGetTilingDefInfo(const char *optype, char *result_info, size_t result_info_len);

ge::graphStatus check_supported_stub(const ge::Operator &op, ge::AscendString &result) {
  const nlohmann::json res_json = R"(
{"ret_code": "1","reason": "check_supported_stub"})"_json;
  std::string res_json_str = res_json.dump();
  result = AscendString(res_json_str.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus op_select_format_stub(const ge::Operator &op, ge::AscendString &result) {
  const nlohmann::json res_json = R"({"op_info": "op_select_format_stub"})"_json;
  std::string res_json_str = res_json.dump();
  result = AscendString(res_json_str.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus get_op_support_info_stub(const ge::Operator &op, ge::AscendString &result) {
  const nlohmann::json res_json = R"({"op_info": "get_op_support_info_stub"})"_json;
  std::string res_json_str = res_json.dump();
  result = AscendString(res_json_str.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus get_op_specific_info_stub(const ge::Operator &op, ge::AscendString &result) {
  const nlohmann::json res_json = R"({"op_info": "get_op_specific_info_stub"})"_json;
  std::string res_json_str = res_json.dump();
  result = AscendString(res_json_str.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus check_supported_stub_throw(const ge::Operator &op, ge::AscendString &result) {
  throw "bad callback";
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus check_supported_stub_fail(const ge::Operator &op, ge::AscendString &result) {
  return ge::GRAPH_FAILED;
}

TEST_F(UtestRegister, ascendC_py_interface_check_cap_ok) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "int32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int32","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_int64","value": [1,2, 3, 4]},
{ "name": "attr_1","dtype": "int","value": 99},
{ "name": "attr_2","dtype": "list_int32","value": [1, 2, 3, 4]},
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  std::string op_type = "ascendC_py_interface_check_cap_ok";
  std::string res_info(100, 'a');
  size_t size = 100;
  // check_supported
  REG_CHECK_SUPPORT(ascendC_py_interface_check_cap_ok, check_supported_stub);
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_CHECK_SUPPORTED, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            1);
  std::string check_supported_result = "{\"reason\":\"check_supported_stub\",\"ret_code\":\"1\"}";
  EXPECT_EQ(check_supported_result, res_info.substr(0, check_supported_result.size()));

  // op_select_format
  REG_OP_SELECT_FORMAT(ascendC_py_interface_check_cap_ok, op_select_format_stub);
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_OP_SELECT_FORMAT, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            1);
  std::string op_select_format_result = "{\"op_info\":\"op_select_format_stub\"}";
  EXPECT_EQ(op_select_format_result, res_info.substr(0, op_select_format_result.size()));

  // get_op_support_info
  REG_OP_SUPPORT_INFO(ascendC_py_interface_check_cap_ok, get_op_support_info_stub);
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_GET_OP_SUPPORT_INFO, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            1);
  std::string get_op_support_info_result = "{\"op_info\":\"get_op_support_info_stub\"}";
  EXPECT_EQ(get_op_support_info_result, res_info.substr(0, get_op_support_info_result.size()));

  // get_op_specific_info
  REG_OP_SPEC_INFO(ascendC_py_interface_check_cap_ok, get_op_specific_info_stub);
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_GET_SPECIFIC_INFO, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            1);
  std::string get_op_specific_info_result = "{\"op_info\":\"get_op_specific_info_stub\"}";
  EXPECT_EQ(get_op_specific_info_result, res_info.substr(0, get_op_specific_info_result.size()));
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_check_cap_fail_without_callback) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "int32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int32","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_int64","value": [1,2, 3, 4]},
{ "name": "attr_1","dtype": "int","value": 99},
{ "name": "attr_2","dtype": "list_int32","value": [1, 2, 3, 4]},
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  std::string op_type = "ascendC_py_interface_check_cap_fail_without_callback";
  std::string res_info(100, 'a');
  size_t size = 100;
  // check_supported
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_CHECK_SUPPORTED, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);

  // op_select_format
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_OP_SELECT_FORMAT, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);

  // get_op_support_info
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_GET_OP_SUPPORT_INFO, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);

  // get_op_specific_info
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_GET_SPECIFIC_INFO, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_check_cap_fail_throw) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "int32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int32","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_int64","value": [1,2, 3, 4]},
{ "name": "attr_1","dtype": "int","value": 99},
{ "name": "attr_2","dtype": "list_int32","value": [1, 2, 3, 4]},
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  std::string op_type = "ascendC_py_interface_check_cap_fail_throw";
  std::string res_info(100, 'a');
  size_t size = 100;
  // check_supported
  REG_CHECK_SUPPORT(ascendC_py_interface_check_cap_fail_throw, check_supported_stub_throw);
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_CHECK_SUPPORTED, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_check_cap_fail_by_callback) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "int32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int32","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_int64","value": [1,2, 3, 4]},
{ "name": "attr_1","dtype": "int","value": 99},
{ "name": "attr_2","dtype": "list_int32","value": [1, 2, 3, 4]},
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  std::string op_type = "ascendC_py_interface_check_cap_fail_throw";
  std::string res_info(100, 'a');
  size_t size = 100;
  // check_supported
  REG_CHECK_SUPPORT(ascendC_py_interface_check_cap_fail_throw, check_supported_stub_fail);
  EXPECT_EQ(AscendCPyInterfaceCheckOp(FUNC_CHECK_SUPPORTED, op_type.c_str(), input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_check_cap_fail_without_params) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  EXPECT_EQ(AscendCPyInterfaceCheckOp(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0), 0);
  unsetenv("ENABLE_RUNTIME_V2");
}

ge::graphStatus generalize_stub(const ge::Operator &op, const ge::AscendString &generalize_config, ge::AscendString &result) {
  const nlohmann::json res_json = R"({"op_info": "generalize_stub"})"_json;
  std::string res_json_str = res_json.dump();
  result = AscendString(res_json_str.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus generalize_stub_fail(const ge::Operator &op, const ge::AscendString &generalize_config, ge::AscendString &result) {
  return ge::GRAPH_FAILED;
}

ge::graphStatus generalize_stub_throw(const ge::Operator &op, const ge::AscendString &generalize_config, ge::AscendString &result) {
  const nlohmann::json res_json = R"({"op_info": "generalize_stub"})"_json;
  std::string res_json_str = res_json.dump();
  result = AscendString(res_json_str.c_str());
  throw "bad callback";
  return ge::GRAPH_SUCCESS;
}

TEST_F(UtestRegister, ascendC_py_interface_generalize_ok) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "float16", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "float32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int64","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "uint32","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_list_int64","value": [[1, 2], [3, 4]]},
{ "name": "attr_1","dtype": "uint32","value": 99},
{ "name": "attr_2","dtype": "list_list_int32","value": [[1, 2], [3, 4]]},
{ "name": "op_para_size", "dtype": "uint16", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  std::string op_type = "ascendC_py_interface_generalize_ok";
  std::string generalize_config = "keep_rank";
  std::string res_info(100, 'a');
  size_t size = 100;
  // shape generalize
  REG_OP_PARAM_GENERALIZE(ascendC_py_interface_generalize_ok, generalize_stub);
  EXPECT_EQ(AscendCPyInterfaceGeneralized(op_type.c_str(), input_str.c_str(), output_str.c_str(), attrs_str.c_str(),
                                       generalize_config.c_str(), const_cast<char *>(res_info.c_str()), size),
            1);
  std::string result = "{\"op_info\":\"generalize_stub\"}";
  EXPECT_EQ(result, res_info.substr(0, result.size()));

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_generalize_fail_by_callback) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "float16", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "float32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int64","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "uint32","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_list_int64","value": [[1, 2], [3, 4]]},
{ "name": "attr_1","dtype": "uint32","value": 99},
{ "name": "attr_2","dtype": "list_list_int32","value": [[1, 2], [3, 4]]},
{ "name": "op_para_size", "dtype": "uint16", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  std::string op_type = "ascendC_py_interface_generalize_ok";
  std::string generalize_config = "keep_rank";
  std::string res_info(100, 'a');
  size_t size = 100;
  // shape generalize
  REG_OP_PARAM_GENERALIZE(ascendC_py_interface_generalize_ok, generalize_stub_fail);
  EXPECT_EQ(AscendCPyInterfaceGeneralized(op_type.c_str(), input_str.c_str(), output_str.c_str(), attrs_str.c_str(),
                                       generalize_config.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);

  unsetenv("ENABLE_RUNTIME_V2");
}


TEST_F(UtestRegister, ascendC_py_interface_generalize_fail_without_callback) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int8", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "int32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int32","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_int64","value": [1,2, 3, 4]},
{ "name": "attr_1","dtype": "int","value": 99},
{ "name": "attr_2","dtype": "list_int32","value": [1, 2, 3, 4]},
{ "name": "op_para_size", "dtype": "int", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  std::string op_type = "TestReluV2";
  std::string generalize_config = "keep_rank";
  std::string res_info(100, 'a');
  size_t size = 100;
  // shape generalize
  EXPECT_EQ(AscendCPyInterfaceGeneralized(op_type.c_str(), input_str.c_str(), output_str.c_str(), attrs_str.c_str(),
                                       generalize_config.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_generalize_fail_throw) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "float16", "const_value": [1,2,3,4],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "float32","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"},
{"name": "test_2","dtype": "int64","shape": [6,6,6,6],"ori_shape": [6,6,6,6],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "uint32","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  std::string output_str = output.dump();
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_list_int64","value": [[1, 2], [3, 4]]},
{ "name": "attr_1","dtype": "uint32","value": 99},
{ "name": "attr_2","dtype": "list_list_int32","value": [[1, 2], [3, 4]]},
{ "name": "op_para_size", "dtype": "uint16", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();
  std::string op_type = "ascendC_py_interface_generalize_fail_throw";
  std::string generalize_config = "keep_rank";
  std::string res_info(100, 'a');
  size_t size = 100;
  // shape generalize
  REG_OP_PARAM_GENERALIZE(ascendC_py_interface_generalize_fail_throw, generalize_stub_throw);
  EXPECT_EQ(AscendCPyInterfaceGeneralized(op_type.c_str(), input_str.c_str(), output_str.c_str(), attrs_str.c_str(),
                                       generalize_config.c_str(), const_cast<char *>(res_info.c_str()), size),
            0);

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_generalize_fail_without_params) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  EXPECT_EQ(AscendCPyInterfaceGeneralized(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0), 0);
  unsetenv("ENABLE_RUNTIME_V2");
}

BEGIN_TILING_DATA_DEF(TestMaxPoolTilingData)
// format: TILING_DATA_FIELD_DEF(data_type, field_name);
TILING_DATA_FIELD_DEF(int8_t, dim_0);
TILING_DATA_FIELD_DEF(int16_t, dim_1);
TILING_DATA_FIELD_DEF(int32_t, dim_2);
TILING_DATA_FIELD_DEF(int64_t, dim_3);
TILING_DATA_FIELD_DEF(uint8_t, dim_4);
TILING_DATA_FIELD_DEF(uint16_t, dim_5);
TILING_DATA_FIELD_DEF(uint32_t, dim_6);
TILING_DATA_FIELD_DEF(uint64_t, dim_7);
TILING_DATA_FIELD_DEF(int32_t, act_core_num);
END_TILING_DATA_DEF

// register class
REGISTER_TILING_DATA_CLASS(TestMaxPool, TestMaxPoolTilingData)

BEGIN_TILING_DATA_DEF(TestMaxPoolTilingDataStruct)
// format: TILING_DATA_FIELD_DEF(data_type, field_name);
TILING_DATA_FIELD_DEF_ARR(int8_t, 8, dim_0);
TILING_DATA_FIELD_DEF_STRUCT(TestMaxPoolTilingData, dim_1);
END_TILING_DATA_DEF

// register class
REGISTER_TILING_DATA_CLASS(TestMaxPoolStruct, TestMaxPoolTilingDataStruct)

TEST_F(UtestRegister, ascendC_py_interface_get_tiling_def_ok) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "TestMaxPool";
  std::string res_info(1024, 'a');
  size_t size = 1024;
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(op_type.c_str(), const_cast<char *>(res_info.c_str()), size), 1);
  const nlohmann::json result =
      R"({"class_name":"TestMaxPoolTilingData","data_size":40,"fields":[{"classType":"0","dtype":"int8_t","name":"dim_0"},{"arrSize":1,"classType":"1","dtype":"uint8_t","name":"dim_1PH"},{"classType":"0","dtype":"int16_t","name":"dim_1"},{"classType":"0","dtype":"int32_t","name":"dim_2"},{"classType":"0","dtype":"int64_t","name":"dim_3"},{"classType":"0","dtype":"uint8_t","name":"dim_4"},{"arrSize":1,"classType":"1","dtype":"uint8_t","name":"dim_5PH"},{"classType":"0","dtype":"uint16_t","name":"dim_5"},{"classType":"0","dtype":"uint32_t","name":"dim_6"},{"classType":"0","dtype":"uint64_t","name":"dim_7"},{"classType":"0","dtype":"int32_t","name":"act_core_num"},{"arrSize":4,"classType":"1","dtype":"uint8_t","name":"TestMaxPoolTilingDataPH"}]})"_json;
  std::string result_str = result.dump();
  EXPECT_EQ(result_str, res_info.substr(0, result_str.size()));
  op_type = "TestMaxPoolStruct";
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(op_type.c_str(), const_cast<char *>(res_info.c_str()), size), 1);
  unsetenv("ENABLE_RUNTIME_V2");
}


namespace test1 {
BEGIN_TILING_DATA_DEF(TestMaxPoolTilingStruct)
TILING_DATA_FIELD_DEF_ARR(int8_t, 5, dim_0);
TILING_DATA_FIELD_DEF_STRUCT(TestMaxPoolTilingData, dim_1);
END_TILING_DATA_DEF
}

namespace test2 {
BEGIN_TILING_DATA_DEF(TestMaxPoolTilingStruct)
TILING_DATA_FIELD_DEF_ARR(int8_t, 5, dim_1);
TILING_DATA_FIELD_DEF_STRUCT(TestMaxPoolTilingData, dim_2);
END_TILING_DATA_DEF
}  //name

namespace test3 {
BEGIN_TILING_DATA_DEF(TestMaxPoolTilingStruct)
TILING_DATA_FIELD_DEF(uint64_t, dim_1);
TILING_DATA_FIELD_DEF_STRUCT(TestMaxPoolTilingData, dim_2);
END_TILING_DATA_DEF
}  //infosize

namespace test4 {
BEGIN_TILING_DATA_DEF(TestMaxPoolTilingStruct)
TILING_DATA_FIELD_DEF_ARR(int8_t, 4, dim_0);
TILING_DATA_FIELD_DEF_STRUCT(TestMaxPoolTilingData, dim_1);
END_TILING_DATA_DEF
}  //arrsize
namespace test5 {
BEGIN_TILING_DATA_DEF(TestMaxPoolTilingStruct)
TILING_DATA_FIELD_DEF_ARR(int8_t, 50, dim_0);
END_TILING_DATA_DEF
} //datasize

std::shared_ptr<TilingDef> Test_api1() {
  return std::make_shared<test1::TestMaxPoolTilingStruct>();
}

std::shared_ptr<TilingDef> Test_api2() {
  return std::make_shared<test2::TestMaxPoolTilingStruct>();
}

std::shared_ptr<TilingDef> Test_api3() {
  return std::make_shared<test3::TestMaxPoolTilingStruct>();
}

std::shared_ptr<TilingDef> Test_api4() {
  return std::make_shared<test4::TestMaxPoolTilingStruct>();
}

std::shared_ptr<TilingDef> Test_api5() {
  return std::make_shared<test5::TestMaxPoolTilingStruct>();
}

TEST_F(UtestRegister, test_register_tiling_data) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "Test_MaxPool";
  std::string res_info(1024, 'a');
  size_t size = 1024;
  CTilingDataClassFactory::GetInstance().RegisterTilingData("Test_MaxPool", Test_api1);
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(op_type.c_str(), const_cast<char *>(res_info.c_str()), size), 1);
  const nlohmann::json result1 =
      R"({"class_name":"TestMaxPoolTilingStruct","data_size":48,"fields":[{"arrSize":5,"classType":"1","dtype":"int8_t","name":"dim_0"},{"arrSize":3,"classType":"1","dtype":"uint8_t","name":"dim_1PH"},{"classType":"2","dtype":"struct","name":"dim_1","structSize":40,"structType":"TestMaxPoolTilingData"}]})"_json;
  std::string result_str1 = result1.dump();
  EXPECT_EQ(result_str1, res_info.substr(0, result_str1.size()));

  CTilingDataClassFactory::GetInstance().RegisterTilingData("Test_MaxPool", Test_api2);
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(op_type.c_str(), const_cast<char *>(res_info.c_str()), size), 1);
  const nlohmann::json result2 =
      R"({"class_name":"TestMaxPoolTilingStruct","data_size":48,"fields":[{"arrSize":5,"classType":"1","dtype":"int8_t","name":"dim_0"},{"arrSize":3,"classType":"1","dtype":"uint8_t","name":"dim_1PH"},{"classType":"2","dtype":"struct","name":"dim_1","structSize":40,"structType":"TestMaxPoolTilingData"}]})"_json;
  std::string result_str2 = result2.dump();
  EXPECT_EQ(result_str2, res_info.substr(0, result_str2.size()));

  CTilingDataClassFactory::GetInstance().RegisterTilingData("Test_MaxPool", Test_api3);
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(op_type.c_str(), const_cast<char *>(res_info.c_str()), size), 1);
  const nlohmann::json result3 =
      R"({"class_name":"TestMaxPoolTilingStruct","data_size":48,"fields":[{"arrSize":5,"classType":"1","dtype":"int8_t","name":"dim_0"},{"arrSize":3,"classType":"1","dtype":"uint8_t","name":"dim_1PH"},{"classType":"2","dtype":"struct","name":"dim_1","structSize":40,"structType":"TestMaxPoolTilingData"}]})"_json;
  std::string result_str3 = result3.dump();
  EXPECT_EQ(result_str3, res_info.substr(0, result_str3.size()));

  CTilingDataClassFactory::GetInstance().RegisterTilingData("Test_MaxPool", Test_api4);
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(op_type.c_str(), const_cast<char *>(res_info.c_str()), size), 1);
  const nlohmann::json result4 =
      R"({"class_name":"TestMaxPoolTilingStruct","data_size":48,"fields":[{"arrSize":5,"classType":"1","dtype":"int8_t","name":"dim_0"},{"arrSize":3,"classType":"1","dtype":"uint8_t","name":"dim_1PH"},{"classType":"2","dtype":"struct","name":"dim_1","structSize":40,"structType":"TestMaxPoolTilingData"}]})"_json;
  std::string result_str4 = result4.dump();
  EXPECT_EQ(result_str4, res_info.substr(0, result_str4.size()));

  CTilingDataClassFactory::GetInstance().RegisterTilingData("Test_MaxPool", Test_api5);
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(op_type.c_str(), const_cast<char *>(res_info.c_str()), size), 1);
  const nlohmann::json result5 =
      R"({"class_name":"TestMaxPoolTilingStruct","data_size":48,"fields":[{"arrSize":5,"classType":"1","dtype":"int8_t","name":"dim_0"},{"arrSize":3,"classType":"1","dtype":"uint8_t","name":"dim_1PH"},{"classType":"2","dtype":"struct","name":"dim_1","structSize":40,"structType":"TestMaxPoolTilingData"}]})"_json;
  std::string result_str5 = result5.dump();
  EXPECT_EQ(result_str5, res_info.substr(0, result_str5.size()));
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_get_tiling_def_without_callback) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "TestMaxPoolNotExist";
  std::string res_info(1024, 'a');
  size_t size = 1024;
  // check_supported
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(op_type.c_str(), const_cast<char *>(res_info.c_str()), size), 0);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_get_tiling_def_fail_without_params) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  EXPECT_EQ(AscendCPyInterfaceGetTilingDefInfo(nullptr, nullptr, 0), 0);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_register_tilingdata_record_tiling_struct) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  uint32_t ret = TilingDataStructBase::GetInstance().RecordTilingStruct("TestTiling1", "test.cpp", 1);
  ret = TilingDataStructBase::GetInstance().RecordTilingStruct("TestTiling2", "test.h", 1);
  ret = TilingDataStructBase::GetInstance().RecordTilingStruct("TestTiling1", "test.cpp", 2);
  EXPECT_EQ(ret, 0);
  ret = TilingDataStructBase::GetInstance().RecordTilingStruct("TestTiling1", "test.h", 1);
  EXPECT_EQ(ret, 0);
  ret = TilingDataStructBase::GetInstance().RecordTilingStruct("TestTiling2", "test.h", 1);
  EXPECT_EQ(ret, 0);
  ret = TilingDataStructBase::GetInstance().RecordTilingStruct("TestTiling2", "test.h", 2);
  EXPECT_EQ(ret, 0);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_register_tilingdata_base_ok) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  auto params = TestMaxPoolTilingData();
  params.set_dim_0(0);
  params.set_dim_1(10);
  params.set_dim_2(20);
  params.set_dim_3(30);
  params.set_dim_4(40);
  params.set_dim_5(50);
  params.set_dim_6(60);
  params.set_dim_7(70);
  params.set_act_core_num(8);
  uint8_t res_data[1024];
  int offset = 0;
  params.SaveToBuffer((void *) (&res_data), params.GetDataSize());
  EXPECT_EQ(*((int8_t *) (res_data + offset)), params.get_dim_0());
  offset += sizeof(int16_t);
  EXPECT_EQ(*((int16_t *) (res_data + offset)), params.get_dim_1());
  offset += sizeof(int16_t);
  EXPECT_EQ(*((int32_t *) (res_data + offset)), params.get_dim_2());
  offset += sizeof(int32_t);
  EXPECT_EQ(*((int64_t *) (res_data + offset)), params.get_dim_3());
  offset += sizeof(int64_t);
  EXPECT_EQ(*((uint8_t *) (res_data + offset)), params.get_dim_4());
  offset += sizeof(uint16_t);
  EXPECT_EQ(*((uint16_t *) (res_data + offset)), params.get_dim_5());
  offset += sizeof(uint16_t);
  EXPECT_EQ(*((uint32_t *) (res_data + offset)), params.get_dim_6());
  offset += sizeof(uint32_t);
  EXPECT_EQ(*((uint64_t *) (res_data + offset)), params.get_dim_7());
  offset += sizeof(uint64_t);
  EXPECT_EQ(*((int32_t *) (res_data + offset)), params.get_act_core_num());
  offset += sizeof(int32_t);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_register_tilingdata_base_failed) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  auto paramStruct = TestMaxPoolTilingDataStruct();
  int8_t arr[] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t res_data[1024];
  int offset = 0;
  paramStruct.set_dim_0(arr);
  paramStruct.dim_1.set_dim_0(0);
  paramStruct.dim_1.set_dim_1(10);
  paramStruct.dim_1.set_dim_2(20);
  paramStruct.dim_1.set_dim_3(30);
  paramStruct.dim_1.set_dim_4(40);
  paramStruct.dim_1.set_dim_5(50);
  paramStruct.dim_1.set_dim_6(60);
  paramStruct.dim_1.set_dim_7(70);
  paramStruct.dim_1.set_act_core_num(8);
  paramStruct.SaveToBuffer((void *) (&res_data), 1024);

  auto params = TestMaxPoolTilingData((void *) (&res_data));
  params.set_dim_0(0);
  params.set_dim_1(10);
  params.set_dim_2(20);
  params.set_dim_3(30);
  params.set_dim_4(40);
  params.set_dim_5(50);
  params.set_dim_6(60);
  params.set_dim_7(70);
  params.set_act_core_num(8);
  params.SaveToBuffer((void *) (&res_data), 1024);
  EXPECT_EQ(*((int8_t *) (res_data + offset)), params.get_dim_0());
  offset += sizeof(int16_t);
  EXPECT_EQ(*((int16_t *) (res_data + offset)), params.get_dim_1());
  params.SetDataPtr(res_data);
  unsetenv("ENABLE_RUNTIME_V2");
}

extern "C" int AscendCPyInterfaceOpReplay(const char *optype, const char *soc_version, int block_dim,
                                       const char *tiling_data, const char *kernel_name, const char *entry_file,
                                       const char *output_kernel_file, int core_type, int task_ration);

int replay_stub(ReplayFuncParam &param, const int core_typ) {
  return 1;
}

int replay_stub_throw(ReplayFuncParam &param, const int core_typ) {
  throw "bad callback";
  return 1;
}

int replay_stub_invalid_ret(ReplayFuncParam &param, const int core_typ) {
  return 0;
}

TEST_F(UtestRegister, ascendC_py_interface_op_replay_ok) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "ascendC_py_interface_op_replay_ok";
  std::string soc_version = "ascend710";
  int blkdim = 32;
  std::string tilingdata = "\x00\x14\x00\x00\x00\n\x00(\x1e\x00\x00\x00\x00\x00\x00\x00";
  std::string kernel_name = "ascendC_py_interface_op_replay_ok";
  std::string entry_file = "ascendC_py_interface_op_replay_ok_entry_file.h";
  std::string output_kernel_file = "ascendC_py_interface_op_replay_ok_kernel_file.cce";
  int core_type = 0;
  int task_ration = 1;
  REG_REPLAY_FUNC(ascendC_py_interface_op_replay_ok, ascend710, replay_stub);
  EXPECT_EQ(AscendCPyInterfaceOpReplay(op_type.c_str(), soc_version.c_str(), blkdim, tilingdata.c_str(),
                                    kernel_name.c_str(), entry_file.c_str(), output_kernel_file.c_str(), core_type,
                                    task_ration),
            1);

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_op_replay_fail_without_callback) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "ascendC_py_interface_op_replay_fail_without_callback";
  std::string soc_version = "ascend710";
  int blkdim = 32;
  std::string tilingdata = "\x00\x14\x00\x00\x00\n\x00(\x1e\x00\x00\x00\x00\x00\x00\x00";
  std::string kernel_name = "ascendC_py_interface_op_replay_fail_without_callback";
  std::string entry_file = "ascendC_py_interface_op_replay_fail_without_callback_entry_file.h";
  std::string output_kernel_file = "ascendC_py_interface_op_replay_fail_without_callback_kernel_file.cce";
  int core_type = 0;
  int task_ration = 1;
  EXPECT_EQ(AscendCPyInterfaceOpReplay(op_type.c_str(), soc_version.c_str(), blkdim, tilingdata.c_str(),
                                    kernel_name.c_str(), entry_file.c_str(), output_kernel_file.c_str(), core_type,
                                    task_ration),
            0);

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_op_replay_fail_throw) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "ascendC_py_interface_op_replay_fail_throw";
  std::string soc_version = "ascend710";
  int blkdim = 32;
  std::string tilingdata = "\x00\x14\x00\x00\x00\n\x00(\x1e\x00\x00\x00\x00\x00\x00\x00";
  std::string kernel_name = "ascendC_py_interface_op_replay_fail_throw";
  std::string entry_file = "ascendC_py_interface_op_replay_fail_throw_entry_file.h";
  std::string output_kernel_file = "ascendC_py_interface_op_replay_fail_throw_kernel_file.cce";
  int core_type = 0;
  int task_ration = 1;
  REG_REPLAY_FUNC(ascendC_py_interface_op_replay_fail_throw, ascend710, replay_stub_throw);
  EXPECT_EQ(AscendCPyInterfaceOpReplay(op_type.c_str(), soc_version.c_str(), blkdim, tilingdata.c_str(),
                                    kernel_name.c_str(), entry_file.c_str(), output_kernel_file.c_str(), core_type,
                                    task_ration),
            0);

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_op_replay_fail_without_params) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  EXPECT_EQ(AscendCPyInterfaceOpReplay(nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, 0, 1), 0);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_op_replay_invalid_core_type) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "ascendC_py_interface_op_replay_invalid_core_type";
  std::string soc_version = "ascend710";
  int blkdim = 32;
  std::string tilingdata = "\x00\x14\x00\x00\x00\n\x00(\x1e\x00\x00\x00\x00\x00\x00\x00";
  std::string kernel_name = "ascendC_py_interface_op_replay_invalid_core_type";
  std::string entry_file = "ascendC_py_interface_op_replay_invalid_core_type_entry_file.h";
  std::string output_kernel_file = "ascendC_py_interface_op_replay_invalid_core_type_kernel_file.cce";
  int core_type = 4;
  int task_ration = 1;
  REG_REPLAY_FUNC(ascendC_py_interface_op_replay_invalid_core_type, ascend710, replay_stub);
  EXPECT_EQ(AscendCPyInterfaceOpReplay(op_type.c_str(), soc_version.c_str(), blkdim, tilingdata.c_str(),
                                    kernel_name.c_str(), entry_file.c_str(), output_kernel_file.c_str(), core_type,
                                    task_ration),
            0);

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_op_replay_invalid_task_ration) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "ascendC_py_interface_op_replay_invalid_task_ration";
  std::string soc_version = "ascend710";
  int blkdim = 32;
  std::string tilingdata = "\x00\x14\x00\x00\x00\n\x00(\x1e\x00\x00\x00\x00\x00\x00\x00";
  std::string kernel_name = "ascendC_py_interface_op_replay_invalid_task_ration";
  std::string entry_file = "ascendC_py_interface_op_replay_invalid_task_ration_entry_file.h";
  std::string output_kernel_file = "ascendC_py_interface_op_replay_invalid_task_ration_kernel_file.cce";
  int core_type = 0;
  int task_ration = -1;
  REG_REPLAY_FUNC(ascendC_py_interface_op_replay_invalid_task_ration, ascend710, replay_stub);
  EXPECT_EQ(AscendCPyInterfaceOpReplay(op_type.c_str(), soc_version.c_str(), blkdim, tilingdata.c_str(),
                                    kernel_name.c_str(), entry_file.c_str(), output_kernel_file.c_str(), core_type,
                                    task_ration),
            0);

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, ascendC_py_interface_op_replay_invalid_ret) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  std::string op_type = "ascendC_py_interface_op_replay_invalid_ret";
  std::string soc_version = "ascend710";
  int blkdim = 32;
  std::string tilingdata = "\x00\x14\x00\x00\x00\n\x00(\x1e\x00\x00\x00\x00\x00\x00\x00";
  std::string kernel_name = "ascendC_py_interface_op_replay_invalid_ret";
  std::string entry_file = "ascendC_py_interface_op_replay_invalid_ret_entry_file.h";
  std::string output_kernel_file = "ascendC_py_interface_op_replay_invalid_ret_kernel_file.cce";
  int core_type = 1;
  int task_ration = 2;
  REG_REPLAY_FUNC(ascendC_py_interface_op_replay_invalid_ret, ascend710, replay_stub_invalid_ret);
  EXPECT_EQ(AscendCPyInterfaceOpReplay(op_type.c_str(), soc_version.c_str(), blkdim, tilingdata.c_str(),
                                    kernel_name.c_str(), entry_file.c_str(), output_kernel_file.c_str(), core_type,
                                    task_ration),
            0);

  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestRegister, new_optiling_py_interface_with_null_desc_ok) {
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "float", "const_value": [null,2.0,null,null], "const_value_null_desc": ["inf", null, "nan", "-inf"],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "float","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"}])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();

  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_float","value": [null,2.0,null,null],"value_null_desc": ["inf", null, "nan", "-inf"]},
{ "name": "attr_1","dtype": "float","value": null, "value_null_desc": "inf"},
{ "name": "attr_2","dtype": "list_float","value": [1, 2, 3, 4]},
{ "name": "op_para_size", "dtype": "float", "value": 50}])"_json;
  std::string attrs_str = attrs.dump();

  const char *op_type = "TestReluV2";
  const char *cmp_info = "";

  std::string result =
      R"({"aicpu_block_dim":4,"block_dim":2,"clear_atomic":true,"local_memory_size":0,"schedule_mode":0,"tiling_cond":0,"tiling_data":"060708090A","tiling_key":78,"workspaces":[12]})";
  size_t size = result.length();
  std::string runinfo(size, 'a');

  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubNewWithNullDesc;
  op_impl_func->tiling_parse = OpTilingParseStubNew;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs_str.c_str(), const_cast<char *>(runinfo.c_str()), size + 1U, elapse),
            1);

  EXPECT_EQ(result, runinfo);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_const_value_fail_with_different_size) {
  // const_value size is different from const_value_null_desc
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "float", "const_value": [null,2.0,null,null], "const_value_null_desc": ["inf", null, "nan"],"shape": [4,4,4,4],"format": "ND"}])"_json;
  SupportInfNanWithNullDescInvalidTestCase(input, nullptr, nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_const_value_fail_with_type_not_support_inf_nan) {
  // dtype doesn't support inf and nan
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "int32", "const_value": [null,2.0,null,null], "const_value_null_desc": ["inf", null, "nan", "-inf"],"shape": [4,4,4,4],"format": "ND"}])"_json;
  SupportInfNanWithNullDescInvalidTestCase(input, nullptr, nullptr);
}

TEST_F(UtestRegister, new_optiling_py_interface_attr_fail_with_invalid_null_desc_param) {
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "float", "const_value": [null,2.0,null,null], "const_value_null_desc": ["inf", null, "nan", "-inf"],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "float","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"}])"_json;
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  // value_null_desc has invalid param "abc"
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "list_float","value": [null,2.0,null,null],"value_null_desc": ["inf", null, "nan", "abc"]}])"_json;
  SupportInfNanWithNullDescInvalidTestCase(input, output, attrs);
}

TEST_F(UtestRegister, new_optiling_py_interface_attr_fail_with_value_not_null_but_has_null_desc) {
  const nlohmann::json input = R"([
{"name": "test_0","dtype": "float", "const_value": [null,2.0,null,null], "const_value_null_desc": ["inf", null, "nan", "-inf"],"shape": [4,4,4,4],"format": "ND"},
{"name": "test_1","dtype": "float","shape": [5,5,5,5],"ori_shape": [5,5,5,5],"format": "ND","ori_format": "ND"}])"_json;
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;

  // when attr dtype is float, value is not null, but exist value_null_desc
  const nlohmann::json attrs = R"([
{ "name": "attr_0","dtype": "float","value": 2.0, "value_null_desc": null}])"_json;
  SupportInfNanWithNullDescInvalidTestCase(input, output, attrs);
}

TEST_F(UtestRegister, new_optiling_py_interface_ok_with_bf16_data) {
  const nlohmann::json input = R"([
  {
    "name": "t0",
    "dtype": "bfloat16",
    "const_value": [1.1, 2.1, 3.1, 4.1],
    "shape": [4, 4, 4, 4],
    "ori_shape": [4, 4, 4, 4],
    "format": "ND"
  },
  {
    "dtype": "int8",
    "shape": [4, 4, 4, 4],
    "ori_shape": [4, 4, 4, 4],
    "format": "ND"
  }])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
  {
    "name": "y_0",
    "dtype": "int8",
    "shape": [9, 9, 9, 9],
    "ori_shape": [9, 9, 9, 9],
    "format": "ND",
    "ori_format": "ND"
  }])"_json;
  std::string output_str = output.dump();
  const char *op_type = "TestReluV2";
  const char *cmp_info = "";
  size_t size = 161U;
  std::string runinfo(size, 'a');
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([{"name": "op_para_size", "dtype": "int", "value": 50}])"_json;

  const size_t max_tiling_size = 50U;
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubBf16;
  op_impl_func->tiling_parse = OpTilingParseStubV5;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = max_tiling_size;


  EXPECT_EQ(TbeOpTilingPyInterface(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                                   attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse),
            1);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(UtestRegister, GetRawErrorMessage_fail_map) {
  const nlohmann::json input = R"([
{"name": "t0", "dtype": "float16","const_value": [1.1,2.1,3.1,4.1] ,"shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "FRACTAL_Z", "sub_format" :32},
{"dtype": "int8", "shape": [4,4,4,4], "ori_shape":[4,4,4,4],"format": "ND"}
])"_json;
  std::string input_str = input.dump();
  const nlohmann::json output = R"([
{"name": "y_0","dtype": "int8","shape": [9,9,9,9],"ori_shape" :[9,9,9,9],"format": "ND","ori_format":"ND"}])"_json;
  std::string output_str = output.dump();
  const char *op_type = "TestReluV2";
  const char *cmp_info = "";
  std::string runinfo(161, 'a');
  size_t size = 161;
  const char *cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const nlohmann::json attrs = R"([
{ "name": "op_para_size", "dtype": "int", "value": 50}, { "name": "test_name", "dtype": "list_int", "value": [50, 51]}])"_json;
  const nlohmann::json extra_infos = R"([
{ "op_name": "matmul_all_reduce", "rank_size": 1}])"_json;

  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto op_impl_func = space_registry->CreateOrGetOpImpl(op_type);
  op_impl_func->tiling = OpTilingStubV6;
  op_impl_func->tiling_parse = OpTilingParseStubV5;
  op_impl_func->compile_info_creator = CreateCompileInfo;
  op_impl_func->compile_info_deleter = DeleteCompileInfo;
  op_impl_func->max_tiling_data_size = 50;

  const string expect_result = "{\"error_messages\":[{\"errorcode\":\"E10025\",\"errormsg\":{\"errmsg\":\"22\",\"realpath\":\"11\"},\"type\":1}],\"ret_code\":1}";
  REPORT_PREDEFINED_ERR_MSG("E10025", std::vector<const char *>({"realpath", "errmsg"}), std::vector<const char *>({"11", "22"}));
  EXPECT_EQ(std::string(DoOpTilingForCompile(op_type, cmp_info, cmp_info_hash, input_str.c_str(), output_str.c_str(),
                               attrs.dump().c_str(), const_cast<char *>(runinfo.c_str()), size, elapse,
                               extra_infos.dump().c_str())), expect_result);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}
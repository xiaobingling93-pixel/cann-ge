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

#include "macro_utils/dt_public_scope.h"
#include "generator/ge_generator.h"
#include "graph/utils/tensor_utils.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/operator_factory_impl.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/manager/graph_manager.h"
#include "es_ge_test_ops_c.h"
#include "api/gelib/gelib.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "register/ops_kernel_builder_registry.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "framework/common/helper/model_helper.h"
using namespace std;

namespace ge {
namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kOpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kOpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";
graphStatus InferFunctionStub(Operator &op) { return GRAPH_SUCCESS; }
}
const char *const kKernelLibName = "DNN_VM_GE_LOCAL";
class UtestGeGenerator : public testing::Test {
 protected:
  void SetUp() {
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
    mmSetEnv(kEnvName, opp_path.c_str(), 1);

    std::string path_vendors = opp_path + "vendors";
    std::string path_config = path_vendors + "/config.ini";
    system(("mkdir -p " + path_vendors).c_str());
    system(("echo 'load_priority=customize' > " + path_config).c_str());

    std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
    system(("mkdir -p " + inner_proto_path).c_str());
    inner_proto_path += kOpsProto;
    system(("touch " + inner_proto_path).c_str());
    system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

    std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
    system(("mkdir -p " + inner_tiling_path).c_str());
    inner_tiling_path += kOpMaster;
    system(("touch " + inner_tiling_path).c_str());
    system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

    // register infer func of data/netoutput
    OperatorFactoryImpl::RegisterInferShapeFunc("Data", InferFunctionStub);
    OperatorFactoryImpl::RegisterInferShapeFunc("NetOutput", InferFunctionStub);
  }

  void TearDown() {
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/") + 1);
    mmSetEnv(kEnvName, opp_path.c_str(), 1);

    std::string path_vendors = opp_path + "vendors";
    system(("rm -rf " + path_vendors).c_str());
    std::string path_so = opp_path + kInner;
    system(("rm -rf " + path_so).c_str());
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  }

  class FakeOpsKernelInfoStore : public OpsKernelInfoStore {
   public:
    FakeOpsKernelInfoStore(){supported_ = true;};
    bool supported_;

   private:
    Status Initialize(const std::map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
      return supported_;
    };
    void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override {};
  };

  class FakeOpsKernelBuilder : public OpsKernelBuilder {
   public:
    FakeOpsKernelBuilder() = default;
   private:
    Status Initialize(const map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    Status CalcOpRunningParam(Node &node) override {
      return SUCCESS;
    };
    Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
      domi::TaskDef task_def;
      tasks.push_back(task_def);
      return SUCCESS;
    };
  };

  void InitGeLib() {
    map<string, string> options;
    Status ret = ge::GELib::Initialize(options);
    EXPECT_EQ(ret, SUCCESS);
    auto instance_ptr = ge::GELib::GetInstance();
    EXPECT_NE(instance_ptr, nullptr);

    //  SchedulerConf conf;
    SchedulerConf scheduler_conf;
    scheduler_conf.name = kKernelLibName;
    scheduler_conf.cal_engines[kKernelLibName] = std::make_shared<EngineConf>();
    scheduler_conf.cal_engines[kKernelLibName]->name = kKernelLibName;
    scheduler_conf.cal_engines[kKernelLibName]->scheduler_id = kKernelLibName;
    map<string, SchedulerConf> scheduler_confs;
    scheduler_confs["scheduler"] = scheduler_conf;
    instance_ptr->DNNEngineManagerObj().schedulers_[kKernelLibName] = scheduler_conf;

    OpsKernelInfoStorePtr ops_kernel_info_store_ptr = std::make_shared<FakeOpsKernelInfoStore>();
    OpsKernelManager::GetInstance().ops_kernel_store_.emplace(kKernelLibName, ops_kernel_info_store_ptr);
    OpsKernelBuilderPtr fake_builder = std::make_shared<FakeOpsKernelBuilder>();
    OpsKernelBuilderRegistry::GetInstance().kernel_builders_[kKernelLibName] = fake_builder;
    OpInfo op_info;
    op_info.engine = kKernelLibName;
    op_info.opKernelLib = kKernelLibName;
    OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
    ops_kernel_manager.ops_kernel_info_[DATA].emplace_back(op_info);
    ops_kernel_manager.ops_kernel_info_[ADD].emplace_back(op_info);
    ops_kernel_manager.ops_kernel_info_[ADDN].emplace_back(op_info);
    ops_kernel_manager.ops_kernel_info_[NETOUTPUT].emplace_back(op_info);
    ops_kernel_manager.ops_kernel_info_[IDENTITY].emplace_back(op_info);
  }

  void FinalizeGeLib() {
    auto instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr != nullptr) {
      instance_ptr->Finalize();
    }
  }
};

namespace {
ComputeGraphPtr MakeGraph() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data", "Data", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
  auto output = builder.AddNode("output", "NetOutput", 1, 1);
  builder.AddDataEdge(data, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, output, 0);
  // add infer func
  data->GetOpDescBarePtr()->AddInferFunc(InferFunctionStub);
  addn1->GetOpDescBarePtr()->AddInferFunc(InferFunctionStub);
  output->GetOpDescBarePtr()->AddInferFunc(InferFunctionStub);
  return builder.GetGraph();
}
}  // namespace


graphStatus TestFunc(Operator &op) { return 0; }
graphStatus TestFunc1(Operator &op) { return 1; }
TEST_F(UtestGeGenerator, test_infer_format_for_single_op) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("graph_name");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  OperatorFactoryImpl::RegisterInferFormatFunc("Add", TestFunc);
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("add", "add");
  compute_graph->AddNode(op_desc);
  GeGenerator generator;
  EXPECT_EQ(generator.InferFormatForSingleOp(op_desc, graph), SUCCESS);
  shared_ptr<OpDesc> op_desc1 = std::make_shared<OpDesc>("Add", "Add");
  compute_graph->AddNode(op_desc1);
  EXPECT_EQ(generator.InferFormatForSingleOp(op_desc1, graph), SUCCESS);
  OperatorFactoryImpl::RegisterInferFormatFunc("MatMulV2", TestFunc1);
  shared_ptr<OpDesc> op_desc2 = std::make_shared<OpDesc>("MatMulV2", "MatMulV2");
  GeTensorDesc tensor_desc;
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddOutputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddOutputDesc(tensor_desc), GRAPH_SUCCESS);
  compute_graph->AddNode(op_desc2);
  EXPECT_EQ(generator.InferFormatForSingleOp(op_desc2, graph), FAILED);
}

TEST_F(UtestGeGenerator, test_build_single_op_online) {
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  GeGenerator generator;
  generator.Initialize({});
  ModelBufferData model_buffer;
  EXPECT_EQ(generator.BuildSingleOpModel(op_desc, inputs, outputs, ENGINE_AIVECTOR, false, model_buffer), FAILED);
  const vector<GeTensor> inputs1;
  EXPECT_EQ(generator.BuildSingleOpModel(op_desc, inputs1, outputs, ENGINE_AIVECTOR, false, model_buffer), FAILED);
}

TEST_F(UtestGeGenerator, test_graph_manager) {
  GraphManager graph_manager;
  EnginePartitioner graph_partitioner;

  auto root_graph = MakeGraph();
  auto sub_graph = MakeGraph();
  root_graph->AddSubGraph(sub_graph);

  auto sgi = MakeShared<SubGraphInfo>();
  // set engine name
  sgi->SetEngineName("AIcoreEngine");
  sgi->SetSubGraph(sub_graph);

  auto sgi_gelocal = MakeShared<SubGraphInfo>();
  // set engine name
  sgi_gelocal->SetEngineName("GELOCAL");
  sgi_gelocal->SetSubGraph(sub_graph);

  graph_partitioner.graph_2_input_subgraph_[root_graph] = sgi_gelocal;
  graph_partitioner.graph_2_subgraph_list_.insert({root_graph, {sgi, sgi_gelocal}});
  graph_partitioner.graph_2_subgraph_list_.insert({sub_graph, {sgi, sgi_gelocal}});
  EXPECT_EQ(graph_manager.ConvertGraphToFile(root_graph, graph_partitioner, "./"), GRAPH_SUCCESS);
}

TEST_F(UtestGeGenerator, test_remove_const) {
  GeGenerator generator;
  GeTensorDesc tensor_desc;
  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = {tensor};
  vector<GeTensor> outputs;
  EXPECT_NO_THROW(generator.RemoveConst(inputs, outputs));
}

TEST_F(UtestGeGenerator, test_generate_online_model) {
  GeTensorDesc tensor_desc;
  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  auto compute_graph = MakeGraph();
  compute_graph->TopologicalSorting();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  GeGenerator generator;
  generator.Initialize({});
  std::string name;
  EXPECT_NE(generator.GenerateOfflineModel(graph, name, inputs), SUCCESS);
}

TEST_F(UtestGeGenerator, test_create_generalized_build_attrs) {
  GeGenerator generator;
  auto ret = generator.Initialize({});
  ASSERT_EQ(ret, SUCCESS);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  ge_root_model->root_graph_ = MakeGraph();
  NodePtr data_node = ge_root_model->root_graph_->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  ASSERT_NE(data_node->GetOpDesc(), nullptr);
  auto in_desc = data_node->GetOpDesc()->MutableInputDesc(0);

  GeTensorDesc tensor_desc(GeShape({1, 2}));
  GeTensor tensor(tensor_desc);

  // 1. input shape all known
  {
    in_desc->SetShape(GeShape({1, 2}));
    in_desc->SetOriginShape(GeShape({1, 2}));
    const vector<GeTensor> inputs = {tensor, tensor};
    const vector<GeTensor> outputs = {tensor};
    const vector<pair<string, string>> inputs_name_type = {{"data", DATA}, {"", CONSTANT}};
    std::vector<NamedAttrs> generalized_attrs;
    ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(generalized_attrs.size(), 0);
  }

  // 2. input and output are empty
  {
    const vector<GeTensor> inputs;
    const vector<GeTensor> outputs;
    const vector<pair<string, string>> inputs_name_type = {{"data", DATA}};
    std::vector<NamedAttrs> generalized_attrs;
    in_desc->SetShape(GeShape({-1, -1}));
    in_desc->SetOriginShape(GeShape({-1, -1}));
    in_desc->SetShapeRange({{1, -1}, {1, -1}});
    in_desc->SetOriginShapeRange({{1, -1}, {1, -1}});
    ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(generalized_attrs.size(), 1);
    std::vector<NamedAttrs> input_res_attrs;
    AttrUtils::GetListNamedAttrs(generalized_attrs.at(0), ATTR_NAME_FUZZ_INPUTS_SUPPORTED_ATTRS, input_res_attrs);
    EXPECT_EQ(input_res_attrs.size(), 1);
    std::vector<NamedAttrs> output_res_attrs;
    AttrUtils::GetListNamedAttrs(generalized_attrs.at(0), ATTR_NAME_FUZZ_OUTPUTS_SUPPORTED_ATTRS, output_res_attrs);
    EXPECT_EQ(output_res_attrs.size(), 0);
    std::vector<NamedAttrs> tensors;
    AttrUtils::GetListNamedAttrs(input_res_attrs.at(0), "tensor", tensors);
    EXPECT_EQ(tensors.size(), 1);
    std::vector<int64_t> shape;
    ret = tensors.at(0).GetItem("shape").GetValue<std::vector<int64_t>>(shape);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(shape, std::vector<int64_t>({-1, -1}));
  }

  // 3. normal
  {
    const vector<GeTensor> inputs = {tensor};
    const vector<GeTensor> outputs = {tensor};
    const vector<pair<string, string>> inputs_name_type = {{"data", DATA}};
    std::vector<NamedAttrs> generalized_attrs;
    in_desc->SetShape(GeShape({-1, -1}));
    in_desc->SetOriginShape(GeShape({-1, -1}));
    in_desc->SetShapeRange({{1, -1}, {1, -1}});
    in_desc->SetOriginShapeRange({{1, -1}, {1, -1}});
    ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(generalized_attrs.size(), 1);
    // check input res attrs
    {
      std::vector<NamedAttrs> input_res_attrs;
      AttrUtils::GetListNamedAttrs(generalized_attrs.at(0), ATTR_NAME_FUZZ_INPUTS_SUPPORTED_ATTRS, input_res_attrs);
      EXPECT_EQ(input_res_attrs.size(), 1);
      std::vector<NamedAttrs> tensors;
      AttrUtils::GetListNamedAttrs(input_res_attrs.at(0), "tensor", tensors);
      EXPECT_EQ(tensors.size(), 1);
      std::vector<int64_t> shape;
      ret = tensors.at(0).GetItem("shape").GetValue<std::vector<int64_t>>(shape);
      EXPECT_EQ(ret, SUCCESS);
      EXPECT_EQ(shape, std::vector<int64_t>({-1, -1}));
      std::vector<std::vector<int64_t>> shape_range;
      ret = tensors.at(0).GetItem("shapeRange").GetValue<std::vector<std::vector<int64_t>>>(shape_range);
      EXPECT_EQ(ret, SUCCESS);
      EXPECT_EQ(shape_range, std::vector<std::vector<int64_t>>({{1, -1}, {1, -1}}));
    }

    // check output res attrs
    {
      GeAttrValue::LIST_NAMED_ATTRS output_res_attrs;
      AttrUtils::GetListNamedAttrs(generalized_attrs.at(0), ATTR_NAME_FUZZ_OUTPUTS_SUPPORTED_ATTRS, output_res_attrs);
      EXPECT_EQ(output_res_attrs.size(), 1);
      GeAttrValue::LIST_NAMED_ATTRS tensors;
      AttrUtils::GetListNamedAttrs(output_res_attrs.at(0), "tensor", tensors);
      EXPECT_EQ(tensors.size(), 1);
      GeAttrValue::LIST_INT shape;
      ret = tensors.at(0).GetItem("shape").GetValue<GeAttrValue::LIST_INT>(shape);
      EXPECT_EQ(ret, SUCCESS);
      EXPECT_EQ(shape, GeAttrValue::LIST_INT({-2}));
    }
  }

  // 4. normal with value
  {
    const vector<GeTensor> inputs = {tensor};
    const vector<GeTensor> outputs = {tensor};
    const vector<pair<string, string>> inputs_name_type = {{"data", DATA}};
    std::vector<NamedAttrs> generalized_attrs;
    in_desc->SetShape(GeShape({-1, -1}));
    in_desc->SetOriginShape(GeShape({-1, -1}));
    in_desc->SetShapeRange({{1, -1}, {1, -1}});
    in_desc->SetOriginShapeRange({{1, -1}, {1, -1}});
    AttrUtils::SetBool(in_desc, ATTR_NAME_VALUE_DEPEND, true);
    AttrUtils::SetTensor(in_desc, ATTR_NAME_VALUE, tensor);
    ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(generalized_attrs.size(), 1);
    std::vector<NamedAttrs> input_res_attrs;
    AttrUtils::GetListNamedAttrs(generalized_attrs.at(0), ATTR_NAME_FUZZ_INPUTS_SUPPORTED_ATTRS, input_res_attrs);
    EXPECT_EQ(input_res_attrs.size(), 1);
    std::vector<NamedAttrs> tensors;
    AttrUtils::GetListNamedAttrs(input_res_attrs.at(0), "tensor", tensors);
    EXPECT_EQ(tensors.size(), 1);
    bool has_value = AttrUtils::HasAttr(tensors.at(0), "value");
    EXPECT_EQ(has_value, true);
  }

  // 5. normal with value range
  {
    const vector<GeTensor> inputs = {tensor};
    const vector<GeTensor> outputs = {tensor};
    const vector<pair<string, string>> inputs_name_type = {{"data", DATA}};
    std::vector<NamedAttrs> generalized_attrs;
    in_desc->SetShape(GeShape({-1, -1}));
    in_desc->SetOriginShape(GeShape({-1, -1}));
    in_desc->SetShapeRange({{1, -1}, {1, -1}});
    in_desc->SetOriginShapeRange({{1, -1}, {1, -1}});
    in_desc->SetValueRange({{1, 256}, {1, 256}});
    ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(generalized_attrs.size(), 1);
    std::vector<NamedAttrs> input_res_attrs;
    AttrUtils::GetListNamedAttrs(generalized_attrs.at(0), ATTR_NAME_FUZZ_INPUTS_SUPPORTED_ATTRS, input_res_attrs);
    EXPECT_EQ(input_res_attrs.size(), 1);
    std::vector<NamedAttrs> tensors;
    AttrUtils::GetListNamedAttrs(input_res_attrs.at(0), "tensor", tensors);
    EXPECT_EQ(tensors.size(), 1);
    std::vector<std::vector<int64_t>> value_range;
    ret = tensors.at(0).GetItem("value_range").GetValue<std::vector<std::vector<int64_t>>>(value_range);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(value_range, std::vector<std::vector<int64_t>>({{1, 256}, {1, 256}}));
  }
}

TEST_F(UtestGeGenerator, CreateGeneralizedBuildAttrs_GeneralizedAttrsIsZero_InputShapeAllknowOutputShapeGeneralized) {
  GeGenerator generator;
  auto ret = generator.Initialize({});
  ASSERT_EQ(ret, SUCCESS);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  ge_root_model->root_graph_ = MakeGraph();
  NodePtr data_node = ge_root_model->root_graph_->FindNode("data");
  NodePtr output_node = ge_root_model->root_graph_->FindNode("output");
  ASSERT_NE(data_node, nullptr);
  ASSERT_NE(data_node->GetOpDesc(), nullptr);
  auto in_desc = data_node->GetOpDesc()->MutableInputDesc(0);
  ASSERT_NE(output_node, nullptr);
  ASSERT_NE(output_node->GetOpDesc(), nullptr);
  auto out_desc = output_node->GetOpDesc()->MutableInputDesc(0);

  GeTensorDesc tensor_desc(GeShape({1, 2}));
  GeTensor tensor(tensor_desc);

  const vector<GeTensor> inputs = {tensor};
  const vector<GeTensor> outputs = {tensor};
  const vector<pair<string, string>> inputs_name_type = {{"data", DATA}};
  std::vector<NamedAttrs> generalized_attrs;
  in_desc->SetShape(GeShape({1, 2}));
  in_desc->SetOriginShape(GeShape({1, 2}));
  in_desc->SetShapeRange({{1, 1}, {2, 2}});
  in_desc->SetOriginShapeRange({{1, 1}, {2, 2}});
  out_desc->SetShape(GeShape({-1}));
  out_desc->SetOriginShape(GeShape({-1}));
  ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(generalized_attrs.size(), 1);
  // check input res attrs
  {
    std::vector<NamedAttrs> input_res_attrs;
    AttrUtils::GetListNamedAttrs(generalized_attrs.at(0), ATTR_NAME_FUZZ_INPUTS_SUPPORTED_ATTRS, input_res_attrs);
    EXPECT_EQ(input_res_attrs.size(), 1);
    std::vector<NamedAttrs> tensors;
    AttrUtils::GetListNamedAttrs(input_res_attrs.at(0), "tensor", tensors);
    EXPECT_EQ(tensors.size(), 1);
    std::vector<int64_t> shape;
    ret = tensors.at(0).GetItem("shape").GetValue<std::vector<int64_t>>(shape);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(shape, std::vector<int64_t>({1, 2}));
    std::vector<std::vector<int64_t>> shape_range;
    ret = tensors.at(0).GetItem("shapeRange").GetValue<std::vector<std::vector<int64_t>>>(shape_range);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(shape_range, std::vector<std::vector<int64_t>>({{1, 1}, {2, 2}}));
  }

  // check output res attrs
  {
    GeAttrValue::LIST_NAMED_ATTRS output_res_attrs;
    AttrUtils::GetListNamedAttrs(generalized_attrs.at(0), ATTR_NAME_FUZZ_OUTPUTS_SUPPORTED_ATTRS, output_res_attrs);
    EXPECT_EQ(output_res_attrs.size(), 1);
    GeAttrValue::LIST_NAMED_ATTRS tensors;
    AttrUtils::GetListNamedAttrs(output_res_attrs.at(0), "tensor", tensors);
    EXPECT_EQ(tensors.size(), 1);
    GeAttrValue::LIST_INT shape;
    ret = tensors.at(0).GetItem("shape").GetValue<GeAttrValue::LIST_INT>(shape);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(shape, GeAttrValue::LIST_INT({-2}));
  }
}

TEST_F(UtestGeGenerator, CreateGeneralizedBuildAttrs_GeneralizedAttrsIsZero_InputOutputShapeAllknow) {
    GeGenerator generator;
    auto ret = generator.Initialize({});
    ASSERT_EQ(ret, SUCCESS);
    GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
    ge_root_model->root_graph_ = MakeGraph();
    NodePtr data_node = ge_root_model->root_graph_->FindNode("data");
    NodePtr output_node = ge_root_model->root_graph_->FindNode("output");
    ASSERT_NE(data_node, nullptr);
    ASSERT_NE(data_node->GetOpDesc(), nullptr);
    auto in_desc = data_node->GetOpDesc()->MutableInputDesc(0);
    ASSERT_NE(output_node, nullptr);
    ASSERT_NE(output_node->GetOpDesc(), nullptr);
    auto out_desc = output_node->GetOpDesc()->MutableInputDesc(0);

    GeTensorDesc tensor_desc(GeShape({1, 2}));
    GeTensor tensor(tensor_desc);

    in_desc->SetShape(GeShape({1, 2}));
    in_desc->SetOriginShape(GeShape({1, 2}));
    out_desc->SetShape(GeShape({1, 2}));
    out_desc->SetOriginShape(GeShape({2}));
    const vector<GeTensor> inputs = {tensor, tensor};
    const vector<GeTensor> outputs = {tensor};
    const vector<pair<string, string>> inputs_name_type = {{"data", DATA}, {"", CONSTANT}};
    std::vector<NamedAttrs> generalized_attrs;
    ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(generalized_attrs.size(), 0);
}

TEST_F(UtestGeGenerator, test_create_generalized_build_attrs_failed) {
  GeGenerator generator;
  auto ret = generator.Initialize({});
  ASSERT_EQ(ret, SUCCESS);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  ge_root_model->root_graph_ = MakeGraph();
  NodePtr data_node = ge_root_model->root_graph_->FindNode("data");
  ASSERT_NE(data_node, nullptr);
  ASSERT_NE(data_node->GetOpDesc(), nullptr);
  auto in_desc = data_node->GetOpDesc()->MutableInputDesc(0);

  GeTensorDesc tensor_desc(GeShape({1, 2}));
  GeTensor tensor(tensor_desc);

  // 1. input size is not same with input nodes
  {
    const vector<GeTensor> inputs = {tensor, tensor, tensor};
    const vector<GeTensor> outputs = {tensor};
    const vector<pair<string, string>> inputs_name_type = {{"data", DATA}, {"", CONSTANT}};
    std::vector<NamedAttrs> generalized_attrs;
    ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
    EXPECT_EQ(ret, INTERNAL_ERROR);
  }

  // 2. missing data node
  {
    const vector<GeTensor> inputs = {tensor};
    const vector<GeTensor> outputs = {tensor};
    const vector<pair<string, string>> inputs_name_type = {{"data_missing", DATA}};
    std::vector<NamedAttrs> generalized_attrs;
    ret = generator.CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, generalized_attrs);
    EXPECT_EQ(ret, INTERNAL_ERROR);
  }
}

TEST_F(UtestGeGenerator, test_build_single_op_online_success) {
  InitGeLib();
  GeShape shape({-2});
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginShape(shape);
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->AddInferFunc(InferFunctionStub);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  GeGenerator generator;
  generator.Initialize({});

  std::map<string, string> options;
  options.emplace("ge.host_env_os", "linux");
  options.emplace("ge.host_env_cpu", "x86_64");
  ge::GetThreadLocalContext().SetGraphOption(options);

  ModelBufferData model_buffer;
  Status ret = generator.BuildSingleOpModel(op_desc, inputs, outputs, "file_name", false);
  EXPECT_EQ(ret, SUCCESS);

  GeShape shape2(std::vector<int64_t>{});
  GeTensorDesc tensor_desc2(shape2);
  GeTensor tensor2(tensor_desc2);
  const vector<GeTensor> inputs2 = { tensor2, tensor2 };
  const vector<GeTensor> outputs2 = { tensor2 };
  ret = generator.BuildSingleOpModel(op_desc, inputs2, outputs2, "file_name", false);

  AttrUtils::SetBool(op_desc, "_AllShape", true);
  generator.BuildSingleOpModel(op_desc, inputs, outputs, "file_name", false);

  options.emplace(JIT_COMPILE, "0");
  ge::GetThreadLocalContext().SetGraphOption(options);
  generator.BuildSingleOpModel(op_desc, inputs, outputs, "file_name", false);
  FinalizeGeLib();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGeGenerator, test_get_single_op_build_stage_graph_success) {
  InitGeLib();
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  GeGenerator generator;
  generator.Initialize({});
  ModelBufferData model_buffer;
  ComputeGraphPtr compute_graph = nullptr;

  Status ret = generator.BuildSingleOpModel(op_desc, inputs, outputs, ENGINE_SYS, false, model_buffer,
                                            GraphStage::GRAPH_STAGE_FUZZ, compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  int64_t graph_stage = static_cast<int64_t>(GraphStage::GRAPH_STAGE_RESERVED);
  bool graph_has_been_added = false;
  // test attr has been cleared
  EXPECT_EQ(AttrUtils::GetInt(compute_graph, kGraphDumpStage, graph_stage), false);
  EXPECT_EQ(AttrUtils::GetBool(compute_graph, ATTR_NAME_GRAPH_HAS_BEEN_ADDED, graph_has_been_added), false);

  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, GenerateInfershapeGraphNull) {
  auto &instance = GeGenerator::GetInstance();
  Graph graph("graph");
  EXPECT_EQ(instance.GenerateInfershapeGraph(graph), PARAM_INVALID);
}

TEST_F(UtestGeGenerator, GenerateInfershapeGraph) {
  auto &instance = GeGenerator::GetInstance();
  instance.Initialize({});
  Graph graph("graph");
  EXPECT_EQ(instance.GenerateInfershapeGraph(graph), GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED);
}

TEST_F(UtestGeGenerator, BuildSingleOpModel) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  instance.Initialize({});
  Graph graph("graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  std::vector<GeTensor> inputs;
  std::vector<GeTensor> outputs;
  OpEngineType engine_type = ENGINE_SYS;
  ModelBufferData model_buff;
  EXPECT_NE(instance.BuildSingleOpModel(op_desc, inputs, outputs, engine_type, model_buff), SUCCESS);
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, GenerateModel) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  options["ge.buildMode"] = BUILD_MODE_TUNING;
  instance.Initialize(options);
  auto compute_graph = MakeGraph();
  compute_graph->TopologicalSorting();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string file_name_prefix = "prefix";
  std::vector<GeTensor> inputs;
  ModelBufferData model;
  bool is_offline = true;
  EXPECT_EQ(instance.GenerateModel(graph, file_name_prefix, inputs, model, is_offline), SUCCESS);
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, BuildSingleOp) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  instance.Initialize(options);
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  GeTensor tensor(tensor_desc);
  vector<GeTensor> inputs = { tensor, tensor };
  vector<GeTensor> outputs = { tensor };
  std::string model_file_name = "online";
  OpEngineType engine_type = ENGINE_AICORE;
  ModelBufferData model_buff;
  ComputeGraphPtr comp_graph;
  bool is_offline = false;
  int32_t compile_flag = 0;
  GraphStage graph_stage = GraphStage::GRAPH_STAGE_FUZZ;
  EXPECT_NE(instance.BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type,
                                   model_buff, comp_graph, is_offline, compile_flag, graph_stage), SUCCESS);
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, CheckEngineTypeSupport) {
  InitGeLib();
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  auto compute_graph = std::make_shared<ComputeGraph>("my_graph");
  auto node = compute_graph->AddNode(op_desc);
  EXPECT_EQ(GeGenerator::CheckEngineTypeSupport(node, ENGINE_SYS), SUCCESS);
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICUBE), SUCCESS);
  OpsKernelManager &ops_kernel_manager = ge::GELib::GetInstance()->OpsKernelManagerObj();
  std::vector<OpInfo> vec;
  ops_kernel_manager.ops_kernel_info_[op_desc->GetType()] = vec;
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_VECTOR), SUCCESS);
  OpInfo op_info;
  vec.emplace_back(op_info);
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_VECTOR), SUCCESS);
  OpInfo oi;
  oi.engine = "AIcoreEngine";
  oi.opKernelLib = "opKernelLib";
  vec.push_back(oi);
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
  ops_kernel_manager.ops_kernel_info_[op_desc->GetType()] = vec;
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
  auto p = std::make_shared<FakeOpsKernelInfoStore>();
  ops_kernel_manager.ops_kernel_store_["opKernelLib"] = p;
  EXPECT_EQ(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
  EXPECT_EQ(node->GetOpDesc()->GetOpEngineName(), "AIcoreEngine");
  EXPECT_EQ(node->GetOpDesc()->GetOpKernelLibName(), "opKernelLib");
  p->supported_ = false;
  ops_kernel_manager.ops_kernel_store_[oi.opKernelLib] = p;
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
  FinalizeGeLib();
  EXPECT_NE(GeGenerator::CheckEngineTypeSupport(node, ENGINE_AICORE), SUCCESS);
}

TEST_F(UtestGeGenerator, BuildSingleOpAttr_unregst_oppath) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  instance.Initialize(options);
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetAttr("_unregst_oppath", AnyValue::CreateFrom<int>(1));
  GeTensor tensor(tensor_desc);
  vector<GeTensor> inputs = { tensor, tensor };
  vector<GeTensor> outputs = { tensor };
  std::string model_file_name = "online";
  OpEngineType engine_type = ENGINE_AICORE;
  ModelBufferData model_buff;
  ComputeGraphPtr comp_graph;
  bool is_offline = false;
  int32_t compile_flag = 0;
  GraphStage graph_stage = GraphStage::GRAPH_STAGE_FUZZ;
  EXPECT_NE(instance.BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type,
                                   model_buff, comp_graph, is_offline, compile_flag, graph_stage), SUCCESS);
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, BuildSingleOpOpInfo) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  instance.Initialize(options);
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->AddInferFunc(InferFunctionStub);
  GeTensor tensor(tensor_desc);
  vector<GeTensor> inputs = { tensor, tensor };
  vector<GeTensor> outputs = { tensor };
  std::string model_file_name = "online";
  OpEngineType engine_type = ENGINE_AICORE;
  ModelBufferData model_buff;
  ComputeGraphPtr comp_graph;
  bool is_offline = false;
  int32_t compile_flag = 0;
  GraphStage graph_stage = GraphStage::GRAPH_STAGE_FUZZ;
  OpsKernelManager &ops_kernel_manager = ge::GELib::GetInstance()->OpsKernelManagerObj();
  std::vector<OpInfo> vec;
  OpInfo oi;
  ops_kernel_manager.ops_kernel_info_[op_desc->GetType()] = vec;

  EXPECT_NE(instance.BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type,
                                   model_buff, comp_graph, is_offline, compile_flag, graph_stage), SUCCESS);
  oi.engine = "AIcoreEngine";
  oi.opKernelLib = "opKernelLib";
  vec.push_back(oi);
  ops_kernel_manager.ops_kernel_info_[op_desc->GetType()] = vec;
  auto p = std::make_shared<FakeOpsKernelInfoStore>();
  ops_kernel_manager.ops_kernel_store_["opKernelLib"] = p;
  EXPECT_EQ(instance.BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type,
                                   model_buff, comp_graph, is_offline, compile_flag, graph_stage), SUCCESS);
  p->supported_ = false;
  ops_kernel_manager.ops_kernel_store_[oi.opKernelLib] = p;
  EXPECT_NE(instance.BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type,
                                   model_buff, comp_graph, is_offline, compile_flag, graph_stage), SUCCESS);
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, BuildSingleOpOpInfoNoLib) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  instance.Initialize(options);
  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  GeTensor tensor(tensor_desc);
  vector<GeTensor> inputs = { tensor, tensor };
  vector<GeTensor> outputs = { tensor };
  std::string model_file_name = "online";
  OpEngineType engine_type = ENGINE_AICORE;
  ModelBufferData model_buff;
  ComputeGraphPtr comp_graph;
  bool is_offline = false;
  int32_t compile_flag = 0;
  GraphStage graph_stage = GraphStage::GRAPH_STAGE_FUZZ;
  OpsKernelManager &ops_kernel_manager = ge::GELib::GetInstance()->OpsKernelManagerObj();
  std::vector<OpInfo> vec;
  OpInfo oi;
  oi.engine = "AIcoreEngine";
  oi.opKernelLib = "";
  vec.push_back(oi);
  ops_kernel_manager.ops_kernel_info_[op_desc->GetType()] = vec;
  EXPECT_NE(instance.BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type,
                                   model_buff, comp_graph, is_offline, compile_flag, graph_stage), SUCCESS);
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, CheckForSingleOp) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  instance.Initialize(options);

  GeTensorDesc tensor_desc;
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  GeTensor tensor(tensor_desc);
  vector<GeTensor> inputs = { tensor };
  vector<GeTensor> outputs = { tensor };
  EXPECT_EQ(instance.CheckForSingleOp(op_desc, inputs, outputs), PARAM_INVALID);
  inputs.push_back(tensor);
  outputs.push_back(tensor);
  EXPECT_EQ(instance.CheckForSingleOp(op_desc, inputs, outputs), PARAM_INVALID);
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, GenerateModelAndDumpBuildGraph) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  options["ge.tuningPath"] = "./after_build.txt";
  instance.Initialize(options);
  auto compute_graph = MakeGraph();
  compute_graph->TopologicalSorting();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string file_name_prefix = "prefix";
  std::vector<GeTensor> inputs;
  ModelBufferData model;
  bool is_offline = true;
  std::map<std::string, std::string> graph_option = GetThreadLocalContext().GetAllGraphOptions();
  GetThreadLocalContext().SetGraphOption(graph_option);
  EXPECT_EQ(instance.GenerateModel(graph, file_name_prefix, inputs, model, is_offline), SUCCESS);
  GetThreadLocalContext().SetGraphOption(graph_option);
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, GenerateFlowModelSetOmSysInfo_fail) {
  const auto back_up = GEThreadLocalContext().GetAllGraphOptions();
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = "invalid_version";
  options[ge::FRAMEWORK_TYPE] = "1";
  instance.Initialize(options);
  auto compute_graph = MakeGraph();
  compute_graph->TopologicalSorting();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string file_name_prefix = "prefix";
  std::vector<GeTensor> inputs;
  ModelBufferData model;
  bool is_offline = false;
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(instance.GenerateModel(graph, file_name_prefix, inputs, model, is_offline), SUCCESS);

  ModelData model_data;
  model_data.model_data = model.data.get();
  model_data.model_len = model.length;
  ModelHelper model_helper;
  model_helper.LoadModel(model_data);
  const auto &ge_model = model_helper.GetGeModel();
  EXPECT_NE(ge_model, nullptr);
  std::string soc_version;
  std::string arch_type;
  AttrUtils::GetStr(*ge_model, "soc_version", soc_version);
  AttrUtils::GetStr(*ge_model, "arch_type", arch_type);
  EXPECT_EQ(soc_version, "invalid_version");
  EXPECT_EQ(arch_type, "");
  EXPECT_STREQ(ge_model->GetName().c_str(), compute_graph->GetName().c_str());
  EXPECT_EQ(instance.Finalize(), SUCCESS);
  FinalizeGeLib();
  ge::GEThreadLocalContext().SetGraphOption(back_up);
}

TEST_F(UtestGeGenerator, test_build_single_op_aicpu_dynamic_online) {
  InitGeLib();
  auto &instance = GeGenerator::GetInstance();
  std::map<std::string, std::string> options;
  instance.Initialize(options);
  GeShape shape({1, 2, 3, 4});
  GeTensorDesc tensor_desc(shape);
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("graph_name");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  compute_graph->AddNode(op_desc);
  GeGenerator generator;
  op_desc->SetOpEngineName("DNN_VM_AICPU_ASCEND");
  (void)AttrUtils::SetBool(op_desc, ATTR_SINGLE_OP_SCENE, true);
  (void)AttrUtils::SetBool(op_desc, kAttrSupportDynamicShape, true);

  EXPECT_EQ(instance.ResetAiCpuToDynamicShape(compute_graph), SUCCESS);

  EXPECT_EQ(op_desc->GetInputDesc(0).GetShape().IsUnknownShape(), true);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetShape().IsUnknownShape(), true);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetShape().IsUnknownShape(), true);
  std::vector<std::pair<int64_t, int64_t>> actual_origin_shape_range;
  op_desc->GetInputDesc(0).GetOriginShapeRange(actual_origin_shape_range);
  EXPECT_TRUE(actual_origin_shape_range.empty());
  op_desc->GetInputDesc(1).GetOriginShapeRange(actual_origin_shape_range);
  EXPECT_TRUE(actual_origin_shape_range.empty());
  op_desc->GetOutputDesc(0).GetOriginShapeRange(actual_origin_shape_range);
  EXPECT_TRUE(actual_origin_shape_range.empty());
  FinalizeGeLib();
}

TEST_F(UtestGeGenerator, test_build_single_op_online_with_qos_before) {
  InitGeLib();
  GeShape shape({-2});
  GeTensorDesc tensor_desc(shape);
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->AddInferFunc(InferFunctionStub);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  GeGenerator generator;
  generator.Initialize({});
  std::map<string, string> options;
  options.emplace("ge.host_env_os", "linux");
  options.emplace("ge.host_env_cpu", "x86_64");
  ge::GetThreadLocalContext().SetGraphOption(options);
  ModelBufferData model_buffer;
  Status ret = generator.BuildSingleOpModel(op_desc, inputs, outputs, "file_name", false);
  EXPECT_EQ(ret, SUCCESS);

  GeShape shape2(std::vector<int64_t>{});
  GeTensorDesc tensor_desc2(shape2);
  GeTensor tensor2(tensor_desc2);
  const vector<GeTensor> inputs2 = { tensor2, tensor2 };
  const vector<GeTensor> outputs2 = { tensor2 };
  ret = generator.BuildSingleOpModel(op_desc, inputs2, outputs2, "file_name", false);

  AttrUtils::SetBool(op_desc, "_AllShape", true);
  generator.BuildSingleOpModel(op_desc, inputs, outputs, "file_name", false);

  options.emplace(ge::BUILD_STEP, ge::BUILD_STEP_BEFORE_BUILD);
  options.emplace(JIT_COMPILE, "0");
  ge::GetThreadLocalContext().SetGraphOption(options);
  generator.BuildSingleOpModel(op_desc, inputs, outputs, "file_name", false);
  FinalizeGeLib();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGeGenerator, test_build_single_op_online_with_qos_after) {
  InitGeLib();
  GeShape shape({-2});
  GeTensorDesc tensor_desc(shape);
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  GeTensor tensor(tensor_desc);
  const vector<GeTensor> inputs = { tensor, tensor };
  const vector<GeTensor> outputs = { tensor };

  GeGenerator generator;
  generator.Initialize({});
  ModelBufferData model_buffer;

  AttrUtils::SetBool(op_desc, "_AllShape", true);
  generator.BuildSingleOpModel(op_desc, inputs, outputs, "file_name", false);

  std::map<string, string> options;
  options.emplace(ge::BUILD_STEP, ge::BUILD_STEP_AFTER_BUILD);
  ge::GetThreadLocalContext().SetGraphOption(options);
  Status ret = generator.BuildSingleOpModel(op_desc, inputs, outputs, "file_name", false);
  FinalizeGeLib();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGeGenerator, ResetInputOutputShape_netoutputnode_hassamesize) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("graph_name");
  shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("Data", "Data");
  compute_graph->AddNode(op_desc);
  GeGenerator generator;
  shared_ptr<OpDesc> op_desc1 = std::make_shared<OpDesc>("Add", "Add");
  compute_graph->AddNode(op_desc1);
  shared_ptr<OpDesc> op_desc2 = std::make_shared<OpDesc>("NetOutput", "NetOutput");
  GeTensorDesc tensor_desc;
  tensor_desc.SetShape(GeShape({1, -1}));
  tensor_desc.SetOriginShape(GeShape({1, -1}));
  tensor_desc.SetOriginShapeRange({{1, 1}, {1, 16}});
  tensor_desc.SetShapeRange({{1, 1}, {1, 16}});
  EXPECT_EQ(op_desc2->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc2->AddOutputDesc(tensor_desc), GRAPH_SUCCESS);
  std::vector<GeTensor> inputs_dynamic;
  std::vector<GeTensor> outputs_dynamic;
  GeTensor tensor_input(tensor_desc, std::vector<uint8_t>(4));
  inputs_dynamic.emplace_back(tensor_input);
  outputs_dynamic.emplace_back(tensor_input);

  std::vector<std::pair<std::string, std::string>> inputs_name_type;
  compute_graph->AddNode(op_desc2);
  auto ret = generator.ResetInputOutputShape(compute_graph, inputs_name_type, inputs_dynamic, outputs_dynamic);
  EXPECT_EQ(ret, SUCCESS);
  ComputeGraphPtr compute_graph1 = MakeShared<ComputeGraph>("graph_name2");
  compute_graph1->AddNode(op_desc);
  compute_graph1->AddNode(op_desc1);
  GeGenerator generator1;
  shared_ptr<OpDesc> op_desc3 = std::make_shared<OpDesc>("NetOutput", "NetOutput");
  GeTensorDesc tensor_desc1;
  tensor_desc1.SetShape(GeShape({1, -1}));
  tensor_desc1.SetOriginShape(GeShape({1, -1}));
  // tensor_desc1.SetOriginShapeRange({{1, 1}, {1, 16}});
  EXPECT_EQ(op_desc3->AddInputDesc(tensor_desc), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc3->AddOutputDesc(tensor_desc1), GRAPH_SUCCESS);
  std::vector<GeTensor> inputs_dynamic1;
  std::vector<GeTensor> outputs_dynamic1;
  GeTensor tensor_input1(tensor_desc, std::vector<uint8_t>(4));
  inputs_dynamic1.emplace_back(tensor_input1);
  outputs_dynamic1.emplace_back(tensor_input1);

  std::vector<std::pair<std::string, std::string>> inputs_name_type1;
  compute_graph1->AddNode(op_desc3);
  auto ret1 = generator.ResetInputOutputShape(compute_graph1, inputs_name_type1, inputs_dynamic1, outputs_dynamic1);
  EXPECT_EQ(ret1, SUCCESS);
}
}  // namespace ge

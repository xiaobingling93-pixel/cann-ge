/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/generator/ge_generator.h"

#include <atomic>
#include <set>
#include <cstdlib>

#include "analyzer/analyzer.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/model/ge_model.h"
#include "common/op_so_store/op_so_store_utils.h"
#include "base/err_msg.h"
#include "graph/def_types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/helper/model_helper.h"
#include "framework/common/helper/pre_model_helper.h"
#include "framework/common/helper/nano_model_save_helper.h"
#include "framework/common/helper/om_file_helper.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/manager/session_id_manager.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"
#include "graph/operator_factory_impl.h"
#include "graph/opsproto_manager.h"
#include "base/registry/opp_package_utils.h"
#include "register/op_lib_register_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "api/gelib/gelib.h"
#include "common/ge_inner_attrs.h"
#include "ge/ge_api_types.h"
#include "common/checker.h"
#include "graph/utils/op_type_utils.h"
#include "register/custom_pass_helper.h"

namespace {

struct InputNodeInfo {
  const ge::GeTensorDesc tensor_desc;
  int32_t arg_index;
  bool has_input_desc;
  std::pair<std::string, std::string> input_node_name_type;
  InputNodeInfo(const ge::GeTensorDesc &desc, int32_t arg_idx, bool has_desc)
      : tensor_desc(desc),
        arg_index(arg_idx),
        has_input_desc(has_desc) {}
};
const char *const kAttrExcludeEngines = "_exclude_engines";
const char *const kAttrOpType = "op_type";
const char *const kEngineNameDefault = "default";
const char *const kVectorEngine = "VectorEngine";
const char *const kAIcoreEngine = "AIcoreEngine";
const char *const kEngineNameOfAiCpu = "DNN_VM_AICPU_ASCEND";
const char *const kEngineNameOfAiCpuTf = "DNN_VM_AICPU";
const char *const kFileNameSuffix = "online";
const char *const kAicpuAllshape = "_AllShape";
const char *const kShapeGeneralized = "shape_generalized";
const char *const kShapePrecise = "shape_precise";
const char *const kHcomGroups = "hcom_group_names";
const int32_t kDefaultJobId = 0;
const int32_t kFuzzBuildPattern = 1;
constexpr const char_t *kSocVersion = "soc_version";
constexpr const char_t *kFrameWorkType = "framework_type";
constexpr const char_t *kArchTypeLabel = "version";
constexpr const char_t *kArchTypeKey = "Arch_type";
constexpr const char_t *kArchType = "arch_type";

std::map<ge::OpEngineType, std::string> engine_type_map {
    {ge::ENGINE_SYS, kEngineNameDefault},
    {ge::ENGINE_AICORE, kAIcoreEngine},
    {ge::ENGINE_VECTOR, kVectorEngine}};

bool ContainsDynamicInpus(const ge::OpDesc &op_desc) {
  for (auto &tensor_desc : op_desc.GetAllInputsDescPtr()) {
    if (tensor_desc->MutableShape().IsUnknownShape()) {
      GELOGI("Contains unknown shape input. set is_dynamic_input to true.");
      return true;
    }
  }
  return false;
}
// if optional in/out, format is format_reserved and dtype is dt_undefined
bool IsOptional(const ge::GeTensorDesc &tensor_desc) {
  return tensor_desc.GetFormat() == ge::FORMAT_RESERVED && tensor_desc.GetDataType() == ge::DT_UNDEFINED;
}

bool IsMobile()
{
  const std::set<std::string> mobile_soc_version_list = {
    "KirinX90", "Kirin9030",
  };
  std::string soc_version;
  (void)ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);
  GELOGI("[Mobile] SOC_VERSION: %s.", soc_version.c_str());
  if (mobile_soc_version_list.find(soc_version) == mobile_soc_version_list.end()) {
    return false;
  }
  GELOGI("[Mobile] SOC_VERSION: %s is mobile.", soc_version.c_str());
  return true;
}
}  // namespace

namespace ge {
static void CreateGeneralizedTensorAttr(const GeTensorDesc &tensor_desc, size_t input_index,
                                        ge::NamedAttrs &attr) {
  attr.SetName("input" + std::to_string(input_index));
  attr.SetAttr("index", ge::GeAttrValue::CreateFrom<int64_t>(input_index));

  std::vector<ge::NamedAttrs> tensor_attrs;
  ge::NamedAttrs tensor_attr;
  tensor_attr.SetName("tensor");

  // 1. shape
  auto origin_shape = tensor_desc.GetOriginShape().GetDims();
  tensor_attr.SetAttr("shape", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(origin_shape));
  GELOGD("Get origin shape:%s.", tensor_desc.GetOriginShape().ToString().c_str());

  // 2. shape range
  std::vector<std::pair<int64_t, int64_t>> origin_shape_range;
  auto ret = tensor_desc.GetOriginShapeRange(origin_shape_range);
  GELOGD("Get origin shape range ret:%u, size:%zu.", ret, origin_shape_range.size());
  if (ret == GRAPH_SUCCESS && !origin_shape_range.empty()) {
    std::vector<std::vector<int64_t>> range;
    for (const auto &item : origin_shape_range) {
      range.emplace_back(std::vector<int64_t>{item.first, item.second});
    }
    tensor_attr.SetAttr("shapeRange", ge::GeAttrValue::CreateFrom<std::vector<std::vector<int64_t>>>(range));
  }

  // 3. value range or value
  std::vector<std::pair<int64_t, int64_t>> value_range;
  ret = tensor_desc.GetValueRange(value_range);
  if (ret == GRAPH_SUCCESS && !value_range.empty()) {
    std::vector<std::vector<int64_t>> range;
    for (const auto &item : value_range) {
      range.emplace_back(std::vector<int64_t>{item.first, item.second});
    }
    tensor_attr.SetAttr("value_range", ge::GeAttrValue::CreateFrom<std::vector<std::vector<int64_t>>>(range));
  } else {
    bool is_value_depend = false;
    (void)AttrUtils::GetBool(tensor_desc, ATTR_NAME_VALUE_DEPEND, is_value_depend);
    ConstGeTensorPtr tensor_value = nullptr;
    bool has_value = false;
    if (is_value_depend) {
      has_value = AttrUtils::GetTensor(tensor_desc, ATTR_NAME_VALUE, tensor_value);
    }
    if (has_value && tensor_value != nullptr) {
      GeTensor value(*tensor_value);
      tensor_attr.SetAttr("value", ge::GeAttrValue::CreateFrom<ge::GeTensor>(value));
    }
  }

  tensor_attrs.emplace_back(tensor_attr);
  attr.SetAttr("tensor", ge::GeAttrValue::CreateFrom<std::vector<ge::NamedAttrs>>(tensor_attrs));
}

static Status AddInputs(const ComputeGraphPtr &graph, const NodePtr &node, int32_t &data_index,
                        InputNodeInfo &input_node_info) {
  GE_CHECK_NOTNULL_EXEC(graph, return PARAM_INVALID);
  GE_CHECK_NOTNULL_EXEC(node, return PARAM_INVALID);
  input_node_info.input_node_name_type = std::make_pair("", "");
  auto tensor = input_node_info.tensor_desc;
  auto format = tensor.GetFormat();
  auto data_type = tensor.GetDataType();
  if (format == FORMAT_RESERVED && data_type == DT_UNDEFINED) {
    return SUCCESS;
  }

  int32_t index = input_node_info.arg_index;
  std::string op_type;
  bool is_const = false;
  (void)AttrUtils::GetBool(tensor, CONST_ATTR_NAME_INPUT, is_const);
  if (is_const) {
    GELOGD("Get input[%d] is const", index);
    op_type = CONSTANTOP;
  } else if (!AttrUtils::GetStr(tensor, kAttrOpType, op_type) || op_type.empty()) {
    op_type = DATA;
  }

  std::string op_name = node->GetName() + "_in_" + std::to_string(index);
  OpDescPtr data_op = MakeShared<ge::OpDesc>(op_name, op_type);
  if (data_op == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "create OpDesc failed, name:%s", op_name.c_str());
    GELOGE(FAILED, "[Create][OpDesc] failed, name:%s", op_name.c_str());
    return FAILED;
  }
  input_node_info.input_node_name_type = std::make_pair(op_name, op_type);
  if (is_const) {
    ConstGeTensorPtr tensor_value;
    if (!AttrUtils::GetTensor(tensor, ge::ATTR_NAME_WEIGHTS, tensor_value)) {
      REPORT_INNER_ERR_MSG("E19999", "get attr %s failed, tensor:%s.",
                        ge::ATTR_NAME_WEIGHTS.c_str(), tensor.GetName().c_str());
      GELOGE(FAILED, "[Get][Attr] %s failed, tensor:%s.", ge::ATTR_NAME_WEIGHTS.c_str(), tensor.GetName().c_str());
      return FAILED;
    }
    if (!AttrUtils::SetTensor(data_op, ge::ATTR_NAME_WEIGHTS, tensor_value)) {
      REPORT_INNER_ERR_MSG("E19999", "set attr %s failed, op:%s.", ge::ATTR_NAME_WEIGHTS.c_str(), op_name.c_str());
      GELOGE(FAILED, "[Set][Attr] %s failed, op:%s.", ge::ATTR_NAME_WEIGHTS.c_str(), op_name.c_str());
      return FAILED;
    }
  }

  (void)AttrUtils::SetBool(data_op, "_is_single_op", true);
  (void)AttrUtils::SetBool(data_op, ATTR_NAME_IS_ORIGINAL_INPUT, true);

  GE_CHK_BOOL_EXEC(data_op->AddInputDesc(tensor) == GRAPH_SUCCESS,
                   REPORT_INNER_ERR_MSG("E19999", "AddInputDesc failed for node:%s", data_op->GetName().c_str());
                   return FAILED, "[Add][InputDesc] fail for node:%s", data_op->GetName().c_str());
  GE_CHK_BOOL_EXEC(data_op->AddOutputDesc(tensor) == GRAPH_SUCCESS,
                   REPORT_INNER_ERR_MSG("E19999", "AddOutputDesc failed for node:%s", data_op->GetName().c_str());
                   return FAILED, "[Add][OutputDesc] fail for node:%s", data_op->GetName().c_str());
  if (input_node_info.has_input_desc && !is_const) {
    GE_CHK_BOOL_EXEC(AttrUtils::SetInt(data_op, ATTR_NAME_INDEX, data_index),
                     REPORT_INNER_ERR_MSG("E19999", "set attr %s failed for node:%s",
                                       ATTR_NAME_INDEX.c_str(), data_op->GetName().c_str());
                     return FAILED,
                     "[Set][Attr:%s] fail for node:%s", ATTR_NAME_INDEX.c_str(), data_op->GetName().c_str());
    ++data_index;
  }

  ge::NodePtr arg_node = graph->AddNode(data_op);
  GE_CHK_BOOL_EXEC(arg_node != nullptr,
                   REPORT_INNER_ERR_MSG("E19999", "add node:%s to graph:%s failed", data_op->GetName().c_str(),
                                     graph->GetName().c_str());
                   return FAILED, "[Add][Node] Insert Data node:%s fail", data_op->GetName().c_str());

  GE_CHK_STATUS(GraphUtils::AddEdge(arg_node->GetOutDataAnchor(0), node->GetInDataAnchor(index)),
                "[Add][Edge]fail from node:%s to node:%s", data_op->GetName().c_str(), node->GetName().c_str());

  return SUCCESS;
}

static Status AddOutputs(const ComputeGraphPtr &graph, const NodePtr &node, const std::vector<GeTensor> &outputs) {
  OpDescPtr op_desc = MakeShared<ge::OpDesc>(graph->GetName() + "_" + NODE_NAME_NET_OUTPUT, NETOUTPUT);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "create OpDesc failed, graph:%s", graph->GetName().c_str());
    GELOGE(FAILED, "[Create][OpDesc] failed, graph:%s", graph->GetName().c_str());
    return FAILED;
  }
  (void)AttrUtils::SetBool(op_desc, "_is_single_op", true);
  int32_t count = 0;
  std::vector<std::string> userdef_dtypes;
  for (const auto &out_desc : outputs) {
    GeTensorDesc tensor = out_desc.GetTensorDesc();
    TensorUtils::SetInputTensor(tensor, true);
    GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(tensor) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E19999", "AddInputDesc failed for node:%s",
                     op_desc->GetName().c_str());
                     return FAILED, "[Add][InputDesc]fail for node:%s",
                     op_desc->GetName().c_str());

    TensorUtils::SetInputTensor(tensor, false);
    TensorUtils::SetOutputTensor(tensor, true);
    GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(tensor) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E19999", "AddOutputDesc failed for node:%s",
                     op_desc->GetName().c_str());
                     return FAILED, "[Add][OutputDesc]fail for node:%s",
                     op_desc->GetName().c_str());

    userdef_dtypes.emplace_back(std::to_string(count).append(":")
                                .append(TypeUtils::DataTypeToSerialString(tensor.GetDataType())));
    count++;
  }
  GE_ASSERT_TRUE(ge::AttrUtils::SetListStr(op_desc, ATTR_ATC_USER_DEFINE_DATATYPE, userdef_dtypes),
                 "[Set][ListAttr] op %s graph:%u set output node ATTR_ATC_USER_DEFINE_DATATYPE failed",
                 op_desc->GetName().c_str(), graph->GetGraphID());
  GE_CHECK_NOTNULL_EXEC(graph, return PARAM_INVALID);
  ge::NodePtr out_node = graph->AddNode(op_desc);
  GE_CHK_BOOL_EXEC(out_node != nullptr,
                   REPORT_INNER_ERR_MSG("E19999", "add node:%s to graph:%u failed.",
                                     op_desc->GetName().c_str(),
                                     graph->GetGraphID());
                   return FAILED,
                   "[Add][Node:%s]fail in graph:%u", op_desc->GetName().c_str(), graph->GetGraphID());
  GE_CHECK_NOTNULL_EXEC(node, return PARAM_INVALID);
  for (int32_t i = 0; i < count; ++i) {
    GE_CHK_STATUS(GraphUtils::AddEdge(node->GetOutDataAnchor(i), out_node->GetInDataAnchor(i)),
                  "[Add][Edge]fail from node:%s to node:%s",
                  node->GetName().c_str(),
                  out_node->GetName().c_str());
  }

  return SUCCESS;
}

static Status ResetTensorVecShape(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &inputs_dynamic) {
  for (auto input : inputs) {
    auto input_desc = input.GetTensorDesc();
    GeShape shape_ori = input_desc.GetShape();

    GeShape dynamic_shape(UNKNOWN_RANK);
    std::vector<std::pair<int64_t, int64_t>> dynamic_shape_range;

    ge::GeTensor inputTensor;
    ge::GeTensorDesc desc(input_desc);

    bool is_const = false;
    (void)AttrUtils::GetBool(input_desc, CONST_ATTR_NAME_INPUT, is_const);
    if (!is_const) {
      int64_t storage_format = FORMAT_NCHW;
      if (ge::AttrUtils::GetInt(desc, ge::ATTR_NAME_STORAGE_FORMAT, storage_format) &&
          !ge::AttrUtils::SetListInt(desc, ge::ATTR_NAME_STORAGE_SHAPE, UNKNOWN_RANK)) {
        REPORT_INNER_ERR_MSG("E19999", "Set attr ATTR_NAME_STORAGE_SHAPE failed to op:%s.", desc.GetName().c_str());
        GELOGE(FAILED, "[Set][Attr] ATTR_NAME_STORAGE_SHAPE fail.");
        return FAILED;
      }
      desc.SetShape(dynamic_shape);
      desc.SetShapeRange(dynamic_shape_range);
    }

    inputTensor.SetTensorDesc(desc);
    inputs_dynamic.push_back(inputTensor);
  }
  return SUCCESS;
}

static void CreateInputAttrsFromTensorDesc(const std::vector<GeTensorDesc> &input_nodes_tensor_desc,
                                           std::vector<ge::NamedAttrs> &input_attrs) {
  for (size_t i = 0; i < input_nodes_tensor_desc.size(); ++i) {
    ge::NamedAttrs input_attr;
    CreateGeneralizedTensorAttr(input_nodes_tensor_desc[i], i, input_attr);
    input_attrs.emplace_back(input_attr);
  }
}

static void CreateOutputAttrs(const std::vector<GeTensor> &outputs, GeAttrValue::LIST_NAMED_ATTRS &output_attrs) {
  if (outputs.empty()) {
    GELOGI("Output tensor is empty, not create output attr.");
    return;
  }
  // Outputs do not generalize and return default results with the same number as output tensors.
  for (size_t i = 0; i < outputs.size(); ++i) {
    ge::GeAttrValue::NAMED_ATTRS output_attr;
    output_attr.SetName("input" + std::to_string(i));
    output_attr.SetAttr("index", ge::GeAttrValue::CreateFrom<int64_t>(i));
    ge::GeAttrValue::LIST_NAMED_ATTRS tensor_attrs;
    ge::GeAttrValue::NAMED_ATTRS tensor_attr;
    tensor_attr.SetName("tensor");
    // Default result only contains Shape with unknown rank.
    tensor_attr.SetAttr("shape", ge::GeAttrValue::CreateFrom<ge::GeAttrValue::LIST_INT>(UNKNOWN_RANK));
    tensor_attrs.emplace_back(tensor_attr);
    output_attr.SetAttr("tensor", ge::GeAttrValue::CreateFrom<ge::GeAttrValue::LIST_NAMED_ATTRS>(tensor_attrs));
    output_attrs.emplace_back(output_attr);
  }
}

static Status GetInputTensorDesc(const std::vector<GeTensor> &inputs,
                                 const std::vector<std::pair<std::string, std::string>> &inputs_name_type,
                                 std::unordered_map<std::string, NodePtr> &nodes_map,
                                 bool &input_nodes_all_known_shape,
                                 std::vector<GeTensorDesc> &tensors_desc) {
  if (!inputs.empty() && inputs_name_type.size() != inputs.size()) {
    GELOGE(INTERNAL_ERROR, "The size of input tensor is not same with input node, "
                           "input tensor size:%zu, input nodes size:%zu.",
           inputs.size(), inputs_name_type.size());
    return INTERNAL_ERROR;
  }

  size_t idx = 0;
  for (const auto &input_name_type : inputs_name_type) {
    const NodePtr node = nodes_map[input_name_type.first];
    if ((node == nullptr) && OpTypeUtils::IsDataNode(input_name_type.second)) {
      GELOGE(INTERNAL_ERROR, "Missing data node:%s.", input_name_type.second.c_str());
      return INTERNAL_ERROR;
    }
    if (node != nullptr) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      bool is_value_depend = false;
      (void)AttrUtils::GetBool(op_desc, ATTR_NAME_VALUE_DEPEND, is_value_depend);
      auto tensor_desc = op_desc->GetInputDesc(0);
      if (is_value_depend) {
        (void)AttrUtils::SetBool(tensor_desc, ATTR_NAME_VALUE_DEPEND, is_value_depend);
      }
      tensors_desc.emplace_back(tensor_desc);
      if (tensors_desc.back().GetOriginShape().IsUnknownShape()) {
        input_nodes_all_known_shape = false;
      }
    } else if (!inputs.empty()) {
      tensors_desc.emplace_back(inputs[idx].GetTensorDesc());
    }
    ++idx;
  }
  GELOGI("Input nodes are all known shape:%d, tensor desc size:%zu", input_nodes_all_known_shape, tensors_desc.size());
  return SUCCESS;
}

static bool HasShapeRange(const std::vector<GeTensor> &inputs) {
  for (const auto &input : inputs) {
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    (void)input.GetTensorDesc().GetShapeRange(shape_range);
    if (!shape_range.empty()) {
      GELOGD("Has set shape range.");
      return true;
    }
  }
  return false;
}

class GeGenerator::Impl {
 public:
  explicit Impl(OmgContext &omg_context) : omg_context_(omg_context) {}
  ~Impl() = default;

  Status BuildModel(const Graph &graph, const std::vector<GeTensor> &inputs, GeRootModelPtr &ge_root_model);
  Status SaveModel(const std::string &file_name_prefix, GeModelPtr &model, ModelBufferData &model_buff) const;

  Status SaveRootModel(const std::string &file_name_prefix, const GeRootModelPtr &ge_root_model,
                       ModelBufferData &model_buff, OfflineModelFormat om_format = OfflineModelFormat::OM_FORMAT_DEFAULT) const;
  Status SaveParams(GeModelPtr &ge_model, const std::string &type, const std::map<std::string, GeAttrValue> &attrs,
                    const std::vector<GeTensor> &inputs, const std::vector<GeTensor> &outputs);

  Status GenerateInfershapeGraph(const Graph &graph);

  OmgContext &omg_context_;
  GraphManager graph_manager_;
  bool is_offline_ = true;
  bool is_singleop_unregistered_ = false;
  bool is_fuzz_compile_enable_ = false;
  bool jit_compile_ = true;
  std::string build_mode_;
  std::string build_step_;
  static std::mutex mutex_;
  uint64_t session_id_ = UINT64_MAX;
  std::shared_ptr<GraphRebuildStateCtrl> rebuild_ctrl_;

 private:
  Status BuildModelWithGraphId(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                               GeRootModelPtr &ge_root_model, const std::map<std::string, std::string> &options);
  bool SetAtcVersionInfo(AttrHolder &obj) const;
  void SetOmSystemInfo(AttrHolder &obj) const;
  void SetHostEnvOsCpuInfo(const GeRootModelPtr &ge_root_model, AttrHolder &obj) const;
  void SetHcomGroupRanks(AttrHolder &obj) const;
};

Status GeGenerator::Initialize(const std::map<std::string, std::string> &options) {
  return Initialize(options, domi::GetContext());
}

Status GeGenerator::Initialize(const std::map<std::string, std::string> &options, OmgContext &context) {
  impl_ = ge::MakeShared<Impl>(context);
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "create Impl failed.");
    GELOGE(MEMALLOC_FAILED, "[Create][Impl] Make shared failed");
    return MEMALLOC_FAILED;
  }
  GE_ASSERT_GRAPH_SUCCESS(OpLibRegistry::GetInstance().PreProcessForCustomOp());
  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get ops proto path!");
  }
  GELOGI("Get opsproto path is %s", opsproto_path.c_str());
  OpsProtoManager *manager = OpsProtoManager::Instance();
  GE_CHECK_NOTNULL(manager);
  std::map<std::string, std::string> option_tmp;
  option_tmp.emplace(std::pair<std::string, std::string>(string("ge.opsProtoLibPath"), opsproto_path));
  (void)manager->Initialize(option_tmp);
  gert::OppPackageUtils::LoadAllOppPackage();
  GE_ASSERT_SUCCESS(CustomPassHelper::Instance().Load());

  ret = impl_->graph_manager_.Initialize(options);
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_INIT_FAILED, "[Call][Initialize] Graph manager initialize failed.");
    return GE_GENERATOR_GRAPH_MANAGER_INIT_FAILED;
  }

  // get build mode
  auto iter = options.find(BUILD_MODE);
  if (iter != options.end()) {
    impl_->build_mode_ = iter->second;
  }
  // get build step
  iter = options.find(BUILD_STEP);
  if (iter != options.end()) {
    impl_->build_step_ = iter->second;
  }
  return SUCCESS;
}

Status GeGenerator::Finalize() {
  if (impl_ == nullptr) {
    return SUCCESS;
  }
  CustomPassHelper::Instance().Unload();
  Status ret = impl_->graph_manager_.Finalize();
  if (ret != SUCCESS) {
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED, "[Call][Finalize] Graph manager finalize failed.");
    return GE_GENERATOR_GRAPH_MANAGER_FINALIZE_FAILED;
  }
  return SUCCESS;
}

Status GeGenerator::GenerateOfflineModel(const Graph &graph, const std::string &file_name_prefix,
                                         const std::vector<GeTensor> &inputs,
                                         OfflineModelFormat om_format) {
  GELOGI("Start to generate offline model.");
  ModelBufferData model;
  (void)AttrUtils::SetStr(GraphUtilsEx::GetComputeGraph(graph), ATTR_MODEL_FILE_NAME_PREFIX, file_name_prefix);
  return GenerateModel(graph, file_name_prefix, inputs, model, true, om_format);
}

Status GeGenerator::GenerateOnlineModel(const Graph &graph, const std::vector<GeTensor> &inputs,
                                        ModelBufferData &model) {
  return GenerateModel(graph, "online", inputs, model, false);
}

Status GeGenerator::GenerateInfershapeGraph(const Graph &graph) {
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);

  Status ret = impl_->GenerateInfershapeGraph(graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][GenerateInfershapeGraph] Dump infershape json failed");
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "[Call][Finalize] graph_manager finalize fail.");
    }
    return ret;
  }
  GELOGI("Generate infer shape graph success");
  return SUCCESS;
}

std::mutex GeGenerator::Impl::mutex_;

// Set package version information in the model
bool GeGenerator::Impl::SetAtcVersionInfo(AttrHolder &obj) const {
  std::string path_base = GetModelPath();
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);

  std::string version_path = path_base + "version.info";
  std::string version;
  if (!PluginManager::GetVersionFromPath(version_path, version)) {
    // Due to AllInOne requirements, in order to ensure compatibility, you need to find another directory
    version_path = path_base + "../compiler/version.info";
    if (!PluginManager::GetVersionFromPath(version_path, version)) {
      GELOGW("Get atc version information failed!");
      return false;
    }
  }
  // set version info
  if (!ge::AttrUtils::SetStr(obj, ATTR_MODEL_ATC_VERSION, version)) {
    GELOGW("Ge model set atc version failed!");
    return false;
  }
  return true;
}

void GeGenerator::Impl::SetHcomGroupRanks(AttrHolder &obj) const {
  std::string hcom_group_names;
  (void)ge::GetContext().GetOption(ge::OPTION_EXEC_HCOM_GROUPLIST, hcom_group_names);
  (void)ge::AttrUtils::SetStr(obj, kHcomGroups, hcom_group_names);
  GELOGI("Set [%s] for attr hcom_group_names success", hcom_group_names.c_str());
}

void GeGenerator::Impl::SetOmSystemInfo(AttrHolder &obj) const {
  const auto set_model_attr_func = [&obj](const std::string &key, const std::string &val) -> void {
    if (!ge::AttrUtils::SetStr(obj, key, val)) {
      GELOGW("SetStr of [%s][%s] failed.", key.c_str(), val.c_str());
    }
  };

  std::string soc_version;
  (void)ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);
  set_model_attr_func(kSocVersion, soc_version);

  fe::PlatFormInfos plat_form_infos;
  fe::OptionalInfos optional_infos;
  std::string arch_type;
  if (fe::PlatformInfoManager::GeInstance().GetPlatformInfos(soc_version, plat_form_infos, optional_infos) != 0U) {
    GELOGW("Get platform with soc_version:%s failed.", soc_version.c_str());
  } else {
    (void)plat_form_infos.GetPlatformResWithLock(kArchTypeLabel, kArchTypeKey, arch_type);
    set_model_attr_func(kArchType, arch_type);
  }

  std::string framework_type;
  (void)ge::GetContext().GetOption(ge::FRAMEWORK_TYPE, framework_type);
  auto iter = ge::kFwkTypeToStr.find(framework_type);
  if (iter == ge::kFwkTypeToStr.end()) {
    GELOGW("Can not find framework_type in the map.");
  } else {
    set_model_attr_func(kFrameWorkType, iter->second);
  }
  GELOGD("Set os sys info: soc_version[%s], Arch_type[%s], framework_type[%s]", soc_version.c_str(), arch_type.c_str(),
         framework_type.c_str());
}

void GeGenerator::Impl::SetHostEnvOsCpuInfo(const GeRootModelPtr &ge_root_model, AttrHolder &obj) const {
  std::string host_env_os;
  std::string host_env_cpu;
  GetThreadLocalContext().GetOption(OPTION_HOST_ENV_OS, host_env_os);
  GetThreadLocalContext().GetOption(OPTION_HOST_ENV_CPU, host_env_cpu);
  if (OpSoStoreUtils::IsSoBinType(ge_root_model->GetSoInOmFlag(), SoBinType::kSpaceRegistry)) {
    (void)ge::AttrUtils::SetStr(obj, ATTR_MODEL_HOST_ENV_OS, host_env_os);
    (void)ge::AttrUtils::SetStr(obj, ATTR_MODEL_HOST_ENV_CPU, host_env_cpu);
  }
  return;
}

Status GeGenerator::SetModelNameForDump(const GeRootModelPtr &ge_root_model) {
  bool is_unknown_shape = false;
  Status ret = ge_root_model->CheckIsUnknownShape(is_unknown_shape);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Check][IsUnknownShape]Check root model is unknown shape failed, model id:%u",
           ge_root_model->GetModelId());
    REPORT_INNER_ERR_MSG("E19999", "Check root model is unknown shape failed, model id:%u",
                      ge_root_model->GetModelId());
    return FAILED;
  }
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  GeModelPtr model_root = nullptr;
  if (is_unknown_shape) {
    model_root = MakeShared<GeModel>();
    GE_CHECK_NOTNULL(model_root);
    model_root->SetGraph(ge_root_model->GetRootGraph());
    ge_root_model->SetSubgraphInstanceNameToModel(ge_root_model->GetRootGraph()->GetName(), model_root);
  }
  const auto model_name = ge_root_model->GetRootGraph()->GetName();
  std::map<std::string, GeModelPtr> name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GeModelPtr &ge_model = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];
  GE_CHECK_NOTNULL(ge_model);
  ge_model->SetName(model_name);
  GELOGI("Model name is set from the root graph name %s", model_name.c_str());
  return SUCCESS;
}

Status GeGenerator::GenerateModel(const Graph &graph, const std::string &file_name_prefix,
                                  const std::vector<GeTensor> &inputs,
                                  ModelBufferData &model, bool is_offline,
                                  OfflineModelFormat om_format) {
  GeRootModelPtr ge_root_model = nullptr;
  GE_CHECK_NOTNULL_EXEC(impl_, return PARAM_INVALID);
  impl_->is_offline_ = is_offline;
  Status ret = impl_->BuildModel(graph, inputs, ge_root_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][Model] failed, ret:%u.", ret);
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "[Call][Finalize] graph_manager finalize fail.");
    }
    return ret;
  }

  /// BUILD_MODE_TUNING with BUILD_STEP_BEFORE_UB_MATCH no need save model;
  /// BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER no need save model;
  /// BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER_SUB no need save model.
  if ((impl_->build_mode_ == BUILD_MODE_TUNING) &&
      (impl_->build_step_ == BUILD_STEP_BEFORE_UB_MATCH || impl_->build_step_ == BUILD_STEP_AFTER_BUILDER ||
       impl_->build_step_ == BUILD_STEP_AFTER_BUILDER_SUB)) {
    GELOGI("Build mode:%s with step:%s no need SaveModel.",
           impl_->build_mode_.c_str(),
           impl_->build_step_.c_str());
    return SUCCESS;
  }

  GE_CHECK_NOTNULL(ge_root_model);
  // Move weight files from "./tmp_weight" to "om_path + /weight" before generate om file
  if (is_offline) {
    const auto &compute_graph = ge_root_model->GetRootGraph();
    GE_CHECK_NOTNULL(compute_graph);
    GE_CHK_STATUS_RET(FileConstantUtils::ChangeFilePath(compute_graph, file_name_prefix),
                      "Failed to change file path, graph name:%s", compute_graph->GetName().c_str());
  }
  ret = SetModelNameForDump(ge_root_model);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = impl_->SaveRootModel(file_name_prefix, ge_root_model, model, om_format);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Save][RootModel] failed, ret:%u, file:%s", ret, file_name_prefix.c_str());
    if (impl_->graph_manager_.Finalize() != SUCCESS) {
      GELOGE(FAILED, "graph_manager finalize fail.");
    }
    return ret;
  }
  return SUCCESS;
}

namespace {
  bool IsNeedConnectInputOpForSingleOp(GeTensorDesc &tensor_desc) {
    bool is_need = true;
    // format and dtype is all reserved, stand for Optional input. When singleop scene
    if (tensor_desc.GetFormat() == FORMAT_RESERVED && tensor_desc.GetDataType() == DT_UNDEFINED) {
      is_need = false;
    }
    return is_need;
  }

  Status CheckDynamicSupport(GeModelPtr &ge_model, const ComputeGraphPtr &graph) {
    bool support_dynamic = true;
    bool is_dynamic = false;
    for (const auto &node : graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node);
      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if (op_desc->GetOpEngineName() != kAIcoreEngine) {
        continue;
      }
      if (AttrUtils::HasAttr(op_desc, kAttrSupportDynamicShape)) {
        is_dynamic = true;
        (void) AttrUtils::GetBool(op_desc, kAttrSupportDynamicShape, support_dynamic);
        if (!support_dynamic) {
          GELOGW("Node[%s] doesn't support dynamic shape.", node->GetName().c_str());
          break;
        }
      }
    }
    if (is_dynamic) {
      (void) AttrUtils::SetBool(ge_model, kAttrSupportDynamicShape, support_dynamic);
    }
    return SUCCESS;
  }

  bool CheckIfConstInput(const GeTensorDescPtr &input_tensor_desc) {
    bool is_const = false;
    (void)AttrUtils::GetBool(input_tensor_desc, CONST_ATTR_NAME_INPUT, is_const);
    return is_const;
  }

  Status ResetInputTensorShape(OpDescPtr &op_desc, const GeShape &dynamic_shape) {
    GE_CHECK_NOTNULL(op_desc);
    for (size_t i = 0; i < op_desc->GetAllInputsSize(); i++) {
      auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
      GE_CHECK_NOTNULL(input_desc);
      // pass const input
      if (CheckIfConstInput(input_desc)) {
        continue;
      }
      input_desc->SetShape(dynamic_shape);
      input_desc->SetOriginShapeRange({});
    }
    return SUCCESS;
  }

  Status ResetOutputTensorShape(OpDescPtr &op_desc, const GeShape &dynamic_shape) {
    GE_CHECK_NOTNULL(op_desc);
    for (size_t i = 0; i < op_desc->GetAllOutputsDescSize(); i++) {
      auto output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
      GE_CHECK_NOTNULL(output_desc);
      output_desc->SetShape(dynamic_shape);
      output_desc->SetOriginShapeRange({});
    }
    return SUCCESS;
  }

  Status ResetOpShape(OpDescPtr &op_desc) {
    GE_CHECK_NOTNULL(op_desc);
    GeShape dynamic_shape(UNKNOWN_RANK);
    (void)ResetInputTensorShape(op_desc, dynamic_shape);
    (void)ResetOutputTensorShape(op_desc, dynamic_shape);
    return SUCCESS;
  }

  bool IsAllAicpuNodes(const ComputeGraphPtr &graph) {
    for (const auto &node : graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node->GetOpDesc());
      // pass input and output node
      if (OpTypeUtils::IsDataNode(node->GetType()) || (node->GetType() == CONSTANT) ||
          (node->GetType() == CONSTANTOP) || (node->GetType() == NETOUTPUT)) {
        continue;
      }

      // find if there are aicpu nodes.
      auto op_desc = node->GetOpDesc();
      string engine_name = op_desc->GetOpEngineName();
      if (engine_name.empty()) {
        GELOGE(GRAPH_FAILED, "Get engine failed of node[%s].", node->GetName().c_str());
        return GRAPH_FAILED;
      }
      if ((engine_name != kEngineNameOfAiCpu) && (engine_name != kEngineNameOfAiCpuTf)) {
        GELOGD("node name %s, node type %s, engine name of current node is %s, it is not belong to aicpu",
               node->GetName().c_str(), node->GetType().c_str(), engine_name.c_str());
        return false;
      }
    }
    return true;
  }
}

Status GeGenerator::ResetAiCpuToDynamicShape(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    // pass input and output node
    if (node->GetType() == CONSTANT || node->GetType() == CONSTANTOP) {
      continue;
    }

    // reset aicpu shape to unknown shape
    auto op_desc = node->GetOpDesc();
    if (ResetOpShape(op_desc) != SUCCESS) {
      GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "Reset node[%s] dynamic shape failed.", node->GetName().c_str());
      return ge::GE_CLI_GE_NOT_INITIALIZED;
    }
    GELOGD("Reset dynamic aicpu node [%s] shape success!", node->GetName().c_str());
  }
  GELOGD("Reset dynamic aicpu nodes shape of graph [%s] success!", graph->GetName().c_str());
  return SUCCESS;
}

void GeGenerator::RemoveConst(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) {
  for (auto &input : inputs) {
    GeTensorDesc input_desc = input.GetTensorDesc();
    bool is_const = false;
    (void)AttrUtils::GetBool(input_desc, CONST_ATTR_NAME_INPUT, is_const);
    bool is_optional = IsOptional(input_desc);
    if (!is_optional && !is_const) {
      outputs.emplace_back(input);
    }
  }
}

Status GeGenerator::CheckForSingleOp(const OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                     const std::vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL_EXEC(op_desc, return PARAM_INVALID);
  if (!inputs.empty() && (inputs.size() != op_desc->GetAllInputsSize())) {
    REPORT_PREDEFINED_ERR_MSG("E14001", std::vector<const char *>({"opname", "optype", "value", "reason"}),
        std::vector<const char *>({op_desc->GetName().c_str(), op_desc->GetType().c_str(),
        ("inputs size" + FmtToStr(op_desc->GetAllInputsSize())).c_str(),
        ("Input size is not equal to tensor size " + FmtToStr(inputs.size())).c_str()}));
    GELOGE(PARAM_INVALID, "[Check][Param] Tensor size: %zu, op:%s(%s) Inputs size: %zu, not equal",
           inputs.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetAllInputsSize());
    return PARAM_INVALID;
  }
  if (!outputs.empty() && (outputs.size() != op_desc->GetOutputsSize())) {
    REPORT_PREDEFINED_ERR_MSG("E14001", std::vector<const char *>({"opname", "optype", "value", "reason"}),
        std::vector<const char *>({op_desc->GetName().c_str(), op_desc->GetType().c_str(),
        ("outputs size" + FmtToStr(op_desc->GetOutputsSize())).c_str(),
        ("Input size is not equal to tensor size " + FmtToStr(outputs.size())).c_str()}));
    GELOGE(PARAM_INVALID, "[Check][Param] Tensor size: %zu, op:%s(%s) Outputs size: %zu, not equal",
           outputs.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetOutputsSize());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status GeGenerator::InferFormatForSingleOp(const OpDescPtr &op_desc, const Graph &graph) {
  GE_CHECK_NOTNULL(op_desc);
  if (OperatorFactoryImpl::GetInferFormatFunc(op_desc->GetType()) != nullptr) {
    auto node_op = ge::OperatorFactoryImpl::CreateOperator("node_op", op_desc->GetType());
    if (node_op.IsEmpty()) {
      GELOGW("get op from OperatorFactory fail. op type: %s", op_desc->GetType().c_str());
    } else {
      GELOGD("get op from OperatorFactory success. op type: %s", op_desc->GetType().c_str());
      auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
      if (temp_op_desc == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "GetOpDescFromOperator failed, as return nullptr, type:%s",
                           op_desc->GetType().c_str());
        GELOGE(FAILED, "[Get][OpDesc] temp op desc is null, type:%s", op_desc->GetType().c_str());
        return FAILED;
      }
      if (!op_desc->UpdateInputName(temp_op_desc->GetAllInputName())) {
        GELOGW("InferFormatForSingleOp UpdateInputName failed");
      }
      if (!op_desc->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
        GELOGW("InferFormatForSingleOp UpdateOutputName failed");
      }
    }
    node_op.BreakConnect();
  }
  auto comp_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(comp_graph);
  auto node = comp_graph->FindNode(op_desc->GetName());
  GE_CHECK_NOTNULL(node);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  auto ret = OpDescUtilsEx::CallInferFormatFunc(op_desc, op);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "call InferFormatFunc for single op:%s fail",
                       op_desc->GetName().c_str());
    GELOGE(FAILED, "[Call][InferFormatFunc] for single op:%s fail.", op_desc->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

void GeGenerator::SetFuzzCompile(const std::vector<GeTensor> &inputs, int32_t compile_flag) const {
  std::string jit_compile;
  (void)ge::GetContext().GetOption(JIT_COMPILE, jit_compile);
  if (jit_compile == "0") {
    impl_->jit_compile_ = false;
  }
  impl_->is_fuzz_compile_enable_ = (!HasShapeRange(inputs)) && (compile_flag == kFuzzBuildPattern);
}

bool GeGenerator::IsFuzzCompileEnable() const {
  return impl_->is_fuzz_compile_enable_ || !impl_->jit_compile_;
}

void GeGenerator::AddShapeGeneralizedOption(std::map<std::string, std::string> &graph_options) const {
  std::string build_mode = IsFuzzCompileEnable() ? kShapeGeneralized : kShapePrecise;
  graph_options[SHAPE_GENERALIZED_BUILD_MODE] = build_mode;
  GELOGI("Shape generalized build mode is [%s].", build_mode.c_str());
}

void GeGenerator::AddExcludeEnginesOption(const OpDescPtr &op_desc,
                                          std::map<std::string, std::string> &graph_options) {
  std::string exclude_engines;
  AttrUtils::GetStr(op_desc, kAttrExcludeEngines, exclude_engines);
  graph_options[EXCLUDE_ENGINES] = exclude_engines; // always update option value
  GELOGI("Exclude engines is %s.", exclude_engines.c_str());
}

void GeGenerator::ConvertOpInfosToOptions(const OpDescPtr &op_desc) const {
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  AddShapeGeneralizedOption(graph_options);
  AddExcludeEnginesOption(op_desc, graph_options);
  GetThreadLocalContext().SetGraphOption(graph_options);
  auto global_options = GetThreadLocalContext().GetAllGlobalOptions();
  ModelHelper model_helper;
  if (model_helper.GetHardwareInfo(global_options) == SUCCESS) {
    GetThreadLocalContext().SetGlobalOption(global_options);
  }
}

Status GeGenerator::BuildOriginalGraphInfo(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                           const std::vector<GeTensor> &outputs, const std::string &model_file_name,
                                           bool is_offline, GraphStage graph_stage, Graph &graph,
                                           ComputeGraphPtr &compute_graph,
                                           std::vector<std::pair<std::string, std::string>> &inputs_name_type) {
  GELOGD("Inputs size is %zu, outputs size is %zu.", inputs.size(), outputs.size());
  GE_CHECK_NOTNULL(impl_);
  impl_->is_offline_ = is_offline;

  if (CheckForSingleOp(op_desc, inputs, outputs) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][Param] input param is invalid when build single op:%s!",
           op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  OmgContext &omg_context = impl_->omg_context_;
  omg_context.is_dynamic_input = ContainsDynamicInpus(*op_desc);

  if (op_desc->HasAttr(ATTR_NAME_UNREGST_OPPATH)) {
    impl_->is_singleop_unregistered_ = true;
  }

  ConvertOpInfosToOptions(op_desc);
  InOutTensorRef inputs_outputs = {inputs, outputs};

  GE_CHK_STATUS(BuildSingleOpGraph(op_desc, inputs_outputs, model_file_name, graph, inputs_name_type),
                "[Build][Graph] for single op:%s fail.", op_desc->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(InferFormatForSingleOp(op_desc, graph));
  if (model_file_name == kFileNameSuffix) {
    compute_graph = GraphUtilsEx::GetComputeGraph(graph);
    GE_CHECK_NOTNULL(compute_graph);
    (void)AttrUtils::SetInt(compute_graph, kGraphDumpStage, static_cast<int64_t>(graph_stage));
  }

  return SUCCESS;
}

static void IsOutputNodesAllknownShape(const NodePtr &output_node, bool &output_nodes_all_known_shape) {
  if (output_node == nullptr) {
    GELOGI("output_node is nullptr in Is_output_nodes_all_known_shape.");
    return;
  }
  const auto &op_desc = output_node->GetOpDesc();
  for (size_t index = 0U; index < op_desc->GetInputsSize(); ++index) {
    if (op_desc->GetInputDesc(index).GetOriginShape().IsUnknownShape()) {
      output_nodes_all_known_shape = false;
      break;
    }
  }

  GELOGI("Output nodes are all known shape:%d", output_nodes_all_known_shape);
  return;
}

Status GeGenerator::CreateGeneralizedBuildAttrs(const GeRootModelPtr &ge_root_model,
    const std::vector<GeTensor> &inputs,
    const std::vector<GeTensor> &outputs,
    const std::vector<std::pair<std::string, std::string>> &inputs_name_type,
    std::vector<ge::NamedAttrs> &generalized_build_attrs) {
  GELOGD("Start to create attrs, input tensor size:%zu, input node size:%zu.", inputs.size(), inputs_name_type.size());
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  std::unordered_map<std::string, NodePtr> nodes_map;
  NodePtr output_node = nullptr;
  for (const auto &node : ge_root_model->GetRootGraph()->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GELOGD("Single op graph node:%s.", node->GetName().c_str());
    nodes_map[node->GetName()] = node;
    if (node->GetType() == NETOUTPUT) {
      output_node = node;
    }
  }
  bool input_nodes_all_known_shape = true;
  std::vector<GeTensorDesc> tensors_desc;
  Status ret = GetInputTensorDesc(inputs, inputs_name_type, nodes_map, input_nodes_all_known_shape, tensors_desc);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to get tensor desc.");
    return ret;
  }
  /*
    背景：Fill\range算子，输入shape为静态、输出shape为动态，按照之前的逻辑,在此处直接返回，返回全静态shape的OM
    引发重复OM编译。
    临时规避方案：输入shape为全静态的情况下，依然判断输出shape是否为动态；若为输出shape为动态，将输出shape泛化为-2；
    若输出shape为静态，依然返回全静态shape的OM。
  */
  bool output_nodes_all_known_shape = true;
  IsOutputNodesAllknownShape(output_node, output_nodes_all_known_shape);
  if (input_nodes_all_known_shape && output_nodes_all_known_shape) {
    GELOGI("No need to create generalized build attrs.");
    return SUCCESS;
  }

  std::vector<ge::NamedAttrs> input_generalized_attrs;
  CreateInputAttrsFromTensorDesc(tensors_desc, input_generalized_attrs);
  GELOGD("Input generalized build attrs size is %zu.", input_generalized_attrs.size());
  std::vector<ge::NamedAttrs> output_generalized_attrs;
  CreateOutputAttrs(outputs, output_generalized_attrs);
  GELOGD("Output generalized build attrs size is %zu, output tensor size is %zu.",
         output_generalized_attrs.size(), outputs.size());

  std::string performance_mode;
  GetContext().GetOption(PERFORMANCE_MODE, performance_mode);
  bool is_high_performance = (performance_mode == "high");
  GELOGD("Performance mode is %s", is_high_performance ? "high" : "normal");

  ge::NamedAttrs build_res;
  build_res.SetName(ATTR_NAME_FUZZ_BUILD_RES_ATTRS);
  build_res.SetAttr(ATTR_NAME_FUZZ_IS_HIGH_PERFORMANCE_ATTRS,
                    ge::GeAttrValue::CreateFrom<bool>(is_high_performance));
  build_res.SetAttr(ATTR_NAME_FUZZ_INPUTS_SUPPORTED_ATTRS,
                    ge::GeAttrValue::CreateFrom<std::vector<ge::NamedAttrs>>(input_generalized_attrs));
  build_res.SetAttr(ATTR_NAME_FUZZ_OUTPUTS_SUPPORTED_ATTRS,
                    ge::GeAttrValue::CreateFrom<std::vector<ge::NamedAttrs>>(output_generalized_attrs));
  generalized_build_attrs.emplace_back(build_res);
  return SUCCESS;
}

Status GeGenerator::BuildSingleOp(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                  const std::vector<GeTensor> &outputs, const std::string &model_file_name,
                                  OpEngineType engine_type, ModelBufferData &model_buff, ComputeGraphPtr &comp_graph,
                                  bool is_offline, int32_t compile_flag, GraphStage graph_stage) {
  // 0. Save original attributes.
  const OpDescPtr op_desc_tmp = OpDescUtils::CloneOpDesc(op_desc);
  GE_CHECK_NOTNULL(op_desc_tmp);

  (void)AttrUtils::SetBool(op_desc, ATTR_SINGLE_OP_SCENE, true);
  SetFuzzCompile(inputs, compile_flag);

  // 1. Create ComputeGraph.
  Graph graph;
  std::vector<std::pair<std::string, std::string>> inputs_name_type;
  Status ret = GeGenerator::BuildOriginalGraphInfo(op_desc, inputs, outputs, model_file_name, is_offline, graph_stage,
                                                   graph, comp_graph, inputs_name_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Origin][Graph]Build graph info failed!");
    return ret;
  }

  if (model_file_name == kFileNameSuffix) {
    GE_CHECK_NOTNULL(comp_graph);
    auto node = comp_graph->FindNode(op_desc->GetName());
    GE_CHECK_NOTNULL(node);
    // 2. check engine type when compile online
    ret = CheckEngineTypeSupport(node, engine_type);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Check][EngineType]not support node:%s with engine of %d.", node->GetName().c_str(), engine_type);
      return ret;
    }
  }

  GELOGI("ATC parser success in single op build.");

  GeRootModelPtr ge_root_model = nullptr;
  std::vector<GeTensor> data_inputs;
  RemoveConst(inputs, data_inputs);
  GE_CHK_STATUS_RET_NOLOG(impl_->BuildModel(graph, data_inputs, ge_root_model));
  if (comp_graph != nullptr) {
    int64_t graph_stage_tmp = static_cast<int64_t>(GraphStage::GRAPH_STAGE_RESERVED);
    (void)AttrUtils::GetInt(comp_graph, kGraphDumpStage, graph_stage_tmp);
    if (graph_stage_tmp == static_cast<int64_t>(GraphStage::GRAPH_STAGE_FUZZ)) {
      GELOGD("graph_stage:%ld.", graph_stage_tmp);
      return SUCCESS;
    }
  }

  std::map<std::string, GeAttrValue> op_attrs = op_desc_tmp->GetAllAttrs();
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  std::map<std::string, GeModelPtr> name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  if (name_to_ge_model.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "GetSubgraphInstanceNameToModel failed.");
    GELOGE(PARAM_INVALID, "[Get][Name] GetSubgraphInstanceNameToModel is empty.");
    return PARAM_INVALID;
  }
  const ComputeGraphPtr root_graph = ge_root_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  GeModelPtr &ge_model = name_to_ge_model[root_graph->GetName()];
  GE_CHECK_NOTNULL(ge_model);
  (void) AttrUtils::SetStr(ge_model, kAttrNameSingleOpType, op_desc->GetType());
  (void) AttrUtils::SetBool(ge_model, ATTR_SINGLE_OP_SCENE, true);
  GE_CHK_STATUS_RET_NOLOG(CheckDynamicSupport(ge_model, root_graph));
  (void)AttrUtils::SetBool(ge_model, ATTR_NAME_IS_DYNAMIC_MODEL, root_graph->GetGraphUnknownFlag());
  GELOGI("After build model, The opType in op_desc_tmp is [%s], ATTR_NAME_IS_DYNAMIC_MODEL: %d",
         op_desc_tmp->GetType().c_str(), root_graph->GetGraphUnknownFlag());

  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc, kAicpuAllshape, all_shape);
  GELOGD("Node: %s, all_shape is %d, compile_flag is %d.", op_desc->GetName().c_str(), all_shape, compile_flag);
  (void)AttrUtils::SetInt(ge_model, ATTR_NAME_BUILD_MODE, IsFuzzCompileEnable());
  if (all_shape && IsAllAicpuNodes(root_graph)) {
    (void)AttrUtils::SetBool(ge_model, kAicpuAllshape, all_shape);
    GELOGD("Get aicpu all_shape kernel!");
    std::vector<GeTensor> inputs_dynamic;
    std::vector<GeTensor> outputs_dynamic;
    GE_CHK_STATUS_RET_NOLOG(ResetTensorVecShape(inputs, inputs_dynamic));
    GE_CHK_STATUS_RET_NOLOG(ResetTensorVecShape(outputs, outputs_dynamic));
    GE_CHK_STATUS_RET_NOLOG(ResetAiCpuToDynamicShape(root_graph));
    GE_CHK_STATUS_RET_NOLOG(
        impl_->SaveParams(ge_model, op_desc_tmp->GetType(), op_attrs, inputs_dynamic, outputs_dynamic));
  } else if (IsFuzzCompileEnable()) {
    std::vector<NamedAttrs> fuzz_build_attrs;
    if (CreateGeneralizedBuildAttrs(ge_root_model, inputs, outputs, inputs_name_type, fuzz_build_attrs) != SUCCESS) {
      GELOGE(FAILED, "[Get][FuzzRet]Failed to get fuzz build result of %s.", op_desc->GetName().c_str());
      return FAILED;
    }
    if (!fuzz_build_attrs.empty()) {
      GE_CHK_BOOL_EXEC(AttrUtils::SetListNamedAttrs(ge_model, ATTR_NAME_FUZZ_BUILD_RES_ATTRS, fuzz_build_attrs),
                       REPORT_INNER_ERR_MSG("E19999", "Set model:%s(id:%u) attr:%s failed.",
                                         ge_model->GetName().c_str(), ge_model->GetModelId(),
                                         ATTR_NAME_FUZZ_BUILD_RES_ATTRS.c_str());
                       return FAILED, "Set model:%s(id:%u) attr:%s failed.",
                       ge_model->GetName().c_str(), ge_model->GetModelId(), ATTR_NAME_FUZZ_BUILD_RES_ATTRS.c_str());
    }
    GE_CHK_STATUS_RET_NOLOG(impl_->SaveParams(ge_model, op_desc_tmp->GetType(), op_attrs, inputs, outputs));
  } else {
    std::vector<GeTensor> inputs_dynamic(inputs);
    std::vector<GeTensor> outputs_dynamic(outputs);
    GE_CHK_STATUS_RET_NOLOG(ResetInputOutputShape(root_graph, inputs_name_type, inputs_dynamic, outputs_dynamic));
    GE_CHK_STATUS_RET_NOLOG(impl_->SaveParams(ge_model, op_desc_tmp->GetType(), op_attrs,
                                              inputs_dynamic, outputs_dynamic));
  }
  GELOGI("Start save GeModel to Model buffer");
  GE_CHK_STATUS_RET_NOLOG(impl_->SaveRootModel(model_file_name, ge_root_model, model_buff));
  return SUCCESS;
}

Status GeGenerator::ResetOutputShapeRange(const OpDescPtr &op_desc, const size_t index,
                                          std::vector<std::pair<int64_t, int64_t>> &shape_range) {
  GE_CHK_BOOL_RET_STATUS((op_desc->GetInputsSize() == op_desc->GetOutputsSize()), INTERNAL_ERROR, \
                         "Netoutput node inputs des size and outputs des size must same.");
  (void)op_desc->GetOutputDesc(index).GetShapeRange(shape_range);
  if (shape_range.size() == 0U) {
    // if outputdesc shaperange does not exist, use inputdesc shaperange which infer by ge
    GELOGI("Netoutput do not has outputdesc shape range use inputdes shape range.");
    (void)op_desc->GetInputDesc(index).GetShapeRange(shape_range);
  }
  return SUCCESS;
}

Status GeGenerator::ResetInputOutputShape(const ComputeGraphPtr &graph,
                                          const std::vector<std::pair<std::string, std::string>> &inputs_name_type,
                                          std::vector<GeTensor> &inputs_dynamic,
                                          std::vector<GeTensor> &outputs_dynamic) {
  if ((inputs_dynamic.empty()) || (outputs_dynamic.empty())) {
    return SUCCESS;
  }
  std::unordered_map<std::string, NodePtr> nodes_map;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    nodes_map[node->GetName()] = node;
  }

  size_t input_index = 0;
  for (const auto &input_name_type : inputs_name_type) {
    const NodePtr node = nodes_map[input_name_type.first];
    if (node != nullptr) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const auto data_shape = op_desc->GetOutputDesc(0).GetShape();
      const auto &dims = data_shape.GetDims();
      if (dims.empty() || std::all_of(dims.begin(), dims.end(), [](int64_t val) { return val >= 0; })) {
        ++input_index;
        continue;
      }
      std::vector<std::pair<int64_t, int64_t>> dynamic_shape_range;
      (void)op_desc->GetOutputDesc(0).GetShapeRange(dynamic_shape_range);
      GE_CHK_STATUS_RET_NOLOG(ResetTensorDesc(input_index, data_shape, inputs_dynamic, dynamic_shape_range));
    }
    ++input_index;
  }

  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == NETOUTPUT) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      for (size_t index = 0; index < op_desc->GetInputsSize(); ++index) {
        const auto data_shape = op_desc->GetInputDesc(index).GetShape();
        const auto &dims = data_shape.GetDims();
        if (dims.empty()) {
          continue;
        }
        if (std::any_of(dims.begin(), dims.end(), [](int64_t val) { return val == UNKNOWN_DIM_NUM || \
                                                                    val == UNKNOWN_DIM; })) {
          // refresh output shape range for UNKNOW_DIM
          std::vector<std::pair<int64_t, int64_t>> shape_range;
          ResetOutputShapeRange(op_desc, index, shape_range);
          GE_CHK_STATUS_RET_NOLOG(ResetTensorDesc(index, data_shape, outputs_dynamic, shape_range));
        }
      }
    }
  }
  return SUCCESS;
}

Status GeGenerator::ResetTensorDesc(const size_t index, const GeShape &data_shape,
                                    std::vector<GeTensor> &vector_dynamic,
                                    std::vector<std::pair<int64_t, int64_t>> &dynamic_shape_range) {
  if (index >= vector_dynamic.size()) {
    GELOGE(PARAM_INVALID, "vector num is not match.");
    return PARAM_INVALID;
  }
  GeTensorDesc &desc = vector_dynamic[index].MutableTensorDesc();
  int64_t storage_format = FORMAT_NCHW;
  if (ge::AttrUtils::GetInt(desc, ge::ATTR_NAME_STORAGE_FORMAT, storage_format) &&
      !ge::AttrUtils::SetListInt(desc, ge::ATTR_NAME_STORAGE_SHAPE, data_shape.GetDims())) {
    REPORT_INNER_ERR_MSG("E19999", "Set attr ATTR_NAME_STORAGE_SHAPE failed to op:%s.", desc.GetName().c_str());
    GELOGE(FAILED, "[Set][Attr] ATTR_NAME_STORAGE_SHAPE fail.");
    return FAILED;
  }
  desc.SetShape(data_shape);
  desc.SetShapeRange(dynamic_shape_range);

  return SUCCESS;
}

/**
 * @ingroup ge
 * @brief Compiling a single operator into an offline model
 * @param [in] OpDescPtr &op_desc: Operator description info that needs to be compiled into an offline model file
 * @param [in] std::vector<GeTensor> &inputs: Operator input data description information.
 * @param [in] std::vector<GeTensor> &outputs: Operator output data description information.
 * @param [in] const std::string &model_file_name: Offline model filename.
 * @param [in] compile_flag: op build flag from atc
 * @return SUCCESS handle successfully / others handle failed
 */
Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                       const std::vector<GeTensor> &outputs, const std::string &model_file_name,
                                       int32_t compile_flag) {
  GELOGI("Start to build single op offline model, input size: %zu, output size: %zu", inputs.size(), outputs.size());
  ModelBufferData model_buff;
  ComputeGraphPtr compute_graph = nullptr;
  OpEngineType engine_type = ENGINE_SYS;
  Status status = BuildSingleOp(op_desc, inputs, outputs, model_file_name, engine_type, model_buff, compute_graph,
                                true, compile_flag);
  GELOGI("Finish build single offline model, status: %u", status);
  return status;
}

/**
 * @ingroup ge
 * @brief Compiling a single operator into online buffer
 * @param [in] OpDescPtr &op_desc: Operator description info that needs to be compiled into an offline model file
 * @param [in] std::vector<GeTensor> &inputs: Operator input data description information.
 * @param [in] std::vector<GeTensor> &outputs: Operator output data description information.
 * @param [in] engine_type: specific engine.
 * @param [in] compile_flag: op build flag, compile flag by acl
 * @param [out] ModelBufferData &Model_buff: Model_buff: model buffer of the op.
 * @return SUCCESS handle successfully / others handle failed
 */

Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                       const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                       ModelBufferData &model_buff) {
  GELOGI("Start to build single op online, input size: %zu, output size: %zu", inputs.size(), outputs.size());
  ComputeGraphPtr compute_graph = nullptr;
  Status status = BuildSingleOp(op_desc, inputs, outputs, kFileNameSuffix, engine_type, model_buff,
                                compute_graph, false);
  GELOGI("Finish build single online model, status: %u", status);
  return status;
}

Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                       const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                       int32_t compile_flag, ModelBufferData &model_buff) {
  GELOGI("Start to build single op online, input size: %zu, output size: %zu", inputs.size(), outputs.size());
  ComputeGraphPtr compute_graph = nullptr;
  Status status = BuildSingleOp(op_desc, inputs, outputs, kFileNameSuffix, engine_type, model_buff, compute_graph,
                                false, compile_flag);
  GELOGI("Finish build single online model, status: %u", status);
  return status;
}

Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                       const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                       int32_t compile_flag, ModelBufferData &model_buff,
                                       GraphStage graph_stage, ComputeGraphPtr &compute_graph) {
  GELOGI("Start to build single op online, input size: %zu, output size: %zu", inputs.size(), outputs.size());
  Status status = BuildSingleOp(op_desc, inputs, outputs, kFileNameSuffix, engine_type, model_buff, compute_graph,
                                false, compile_flag, graph_stage);

  // clear attr for tuning
  const std::map<std::string, ge::GeAttrValue> &original_attrs = ge::AttrUtils::GetAllAttrs(compute_graph);
  for (auto const &attr_iter : original_attrs) {
    if (compute_graph->DelAttr(attr_iter.first) != GRAPH_SUCCESS) {
      GELOGW("Delete attr failed.");
    }
  }

  GELOGI("Finish build single online model, status: %u", status);
  return status;
}

Status GeGenerator::BuildSingleOpGraph(const OpDescPtr &op_desc, const InOutTensorRef &inputs_outputs,
                                       std::string graph_name, Graph &graph,
                                       std::vector<std::pair<std::string, std::string>> &inputs_name_type) const {
  ge::ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL_EXEC(compute_graph, return INTERNAL_ERROR);

  // 1. Add Node to ComputeGraph.
  NodePtr op_node = compute_graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(op_node, return INTERNAL_ERROR);

  // 2. Create InputData node.
  int32_t arg_index = 0;
  int32_t data_index = 0;
  const std::vector<ge::GeTensor> &inputs = inputs_outputs.first;
  if (inputs.empty()) {
    for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
      GE_CHECK_NOTNULL_EXEC(input_desc, return INTERNAL_ERROR);
      if (!IsNeedConnectInputOpForSingleOp(*input_desc)) {
        continue;
      }
      InputNodeInfo input_node_info(*input_desc, arg_index, false);
      GE_CHK_STATUS_RET_NOLOG(AddInputs(compute_graph, op_node, data_index, input_node_info));
      inputs_name_type.emplace_back(input_node_info.input_node_name_type);
      arg_index++;
    }
  } else {
    for (const auto &in_desc : inputs) {
      InputNodeInfo input_node_info(in_desc.GetTensorDesc(), arg_index, true);
      GE_CHK_STATUS_RET_NOLOG(AddInputs(compute_graph, op_node, data_index, input_node_info));
      inputs_name_type.emplace_back(input_node_info.input_node_name_type);
      arg_index++;
    }
  }

  // 3. Create Output node.
  const std::vector<ge::GeTensor> &outputs = inputs_outputs.second;
  if (!outputs.empty()) {
    GE_CHK_STATUS_RET_NOLOG(AddOutputs(compute_graph, op_node, outputs));
  }

  (void)AttrUtils::SetBool(compute_graph, ATTR_SINGLE_OP_SCENE, true);
  // dump ComputeGraph node.
  compute_graph->Dump();
  graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  return SUCCESS;
}
Status GeGenerator::CheckEngineTypeSupport(const NodePtr &node, OpEngineType engine_type) {
  GE_ASSERT_NOTNULL(node);
  const OpDescPtr &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  if (engine_type == ENGINE_SYS) {
    GELOGI("CheckEngineType: use default engine.");
    return SUCCESS;
  }

  // get op engine name
  std::string op_engine_name;
  auto iter = engine_type_map.find(engine_type);
  if (iter != engine_type_map.end()) {
    op_engine_name = iter->second;
    GELOGI("CheckEngineType: engine type: %d", static_cast<int32_t>(engine_type));
  } else {
    REPORT_PREDEFINED_ERR_MSG(
        "E14001", std::vector<const char *>({"opname", "optype", "value", "reason"}),
        std::vector<const char *>({op_desc->GetName().c_str(), op_desc->GetType().c_str(),
        "engine type", "It only supports default/AIcoreEngine/VectorEngine"}));
    GELOGE(FAILED,
           "[Check][Param] value:%d not support, "
           "only support default/AIcoreEngine/VectorEngine now",
           static_cast<int32_t>(engine_type));
    return FAILED;
  }

  if (op_desc->HasAttr(ATTR_NAME_UNREGST_OPPATH)) {
    op_desc->SetOpEngineName(op_engine_name);
    op_desc->SetOpKernelLibName(op_engine_name);
    return SUCCESS;
  }
  // set op engine name and opkernelLib. when engine support
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    REPORT_INNER_ERR_MSG("E19999", "get gelib failed, as get instance failed or initflag failed.");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] CheckEngineType failed, as get gelib failed.");
    return FAILED;
  }
  auto &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  const auto &op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    REPORT_PREDEFINED_ERR_MSG("E14001", std::vector<const char *>({"opname", "optype", "value", "reason"}),
                              std::vector<const char *>({op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                                                         "optype", "This optype is not registed"}));
    GELOGE(FAILED, "[Get][OpInfo] by op type %s failed.", op_desc->GetType().c_str());
    return FAILED;
  }
  std::string kernel_name;
  for (const auto &it : op_infos) {
    if (it.engine == op_engine_name) {
      kernel_name = it.opKernelLib;
      break;
    }
  }
  if (kernel_name.empty()) {
    REPORT_PREDEFINED_ERR_MSG("E14001", std::vector<const char *>({"opname", "optype", "value", "reason"}),
                              std::vector<const char *>({op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                                                         ("engine name" + FmtToStr(op_engine_name)).c_str(),
                                                         "This optype is not registed"}));
    GELOGE(FAILED, "[Check][Param] Can not find ops kernel, engine name:%s. op:%s(%s)", op_engine_name.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  const auto &kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
  const auto kernel_info_store = kernel_map.find(kernel_name);
  if (kernel_info_store != kernel_map.end()) {
    std::string unsupported_reason;
    if (kernel_info_store->second->CheckSupported(node, unsupported_reason)) {
      op_desc->SetOpEngineName(op_engine_name);
      op_desc->SetOpKernelLibName(kernel_name);
      GELOGI("CheckEngineType:Set OpKernelLibName %s and engine name %s into op_desc %s", kernel_name.c_str(),
             op_engine_name.c_str(), op_desc->GetName().c_str());
      return SUCCESS;
    } else {
      REPORT_PREDEFINED_ERR_MSG(
          "EZ3002", std::vector<const char *>({"optype", "opskernel", "reason"}),
          std::vector<const char *>({op_desc->GetType().c_str(), kernel_name.c_str(), unsupported_reason.c_str()}));
      GELOGE(FAILED, "[Call][CheckSupported] failed, Op type %s of ops kernel %s is unsupported, reason:%s",
             op_desc->GetType().c_str(), kernel_name.c_str(), unsupported_reason.c_str());
      return FAILED;
    }
  } else {
    REPORT_PREDEFINED_ERR_MSG(
        "EZ3003", std::vector<const char *>({"opname", "optype"}),
        std::vector<const char *>({op_desc->GetName().c_str(), op_desc->GetType().c_str()}));
    GELOGE(FAILED,
           "[Check][Param] Can not find any supported ops kernel info store by kernel_name %s,"
           "op type is %s, op name is %s",
           kernel_name.c_str(), op_desc->GetType().c_str(), op_desc->GetName().c_str());
  }
  return FAILED;
}

Status GeGenerator::SetCurrentSessionId(const uint64_t session_id) const {
  GE_ASSERT_NOTNULL(impl_);
  impl_->session_id_ = session_id;
  return SUCCESS;
}

Status GeGenerator::SetExternalGraphRebuildStateCtrl(void *rebuild_ctrl) const {
  GE_ASSERT_NOTNULL(impl_);
  impl_->rebuild_ctrl_.reset(PtrToPtr<void, GraphRebuildStateCtrl>(rebuild_ctrl),
      [] (const GraphRebuildStateCtrl *rebuild_ctrl_param) { (void)rebuild_ctrl_param; GELOGI("no delete rebuild"); });
  impl_->graph_manager_.SetExternalGraphRebuildStateCtrl(impl_->rebuild_ctrl_);
  return SUCCESS;
}

Status GeGenerator::Impl::SaveParams(GeModelPtr &ge_model, const std::string &type,
                                     const std::map<std::string, GeAttrValue> &attrs,
                                     const std::vector<GeTensor> &inputs, const std::vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL_EXEC(ge_model, return PARAM_INVALID);
  if (graph_manager_.SaveParams(*ge_model, type, attrs, inputs, outputs) != SUCCESS) {
    (void)graph_manager_.Finalize();
    return FAILED;
  }

  return SUCCESS;
}

Status GeGenerator::Impl::SaveModel(const std::string &file_name_prefix, GeModelPtr &model,
                                    ModelBufferData &model_buff) const {
  // set atc version
  if (!SetAtcVersionInfo(*(model.get()))) {
    GELOGW("SetPackageVersionInfo of atc failed!");
  }
  ModelHelper model_helper;
  model_helper.SetSaveMode(is_offline_);

  // Configure attribute compression mode from options
  std::map<std::string, std::string> options;
  std::string attr_compression_value;
  if (GetContext().GetOption(ENABLE_ATTR_COMPRESSION, attr_compression_value) == SUCCESS) {
    (void)model_helper.ConfigureAttrCompressionMode(attr_compression_value);
  }

  Status ret = model_helper.SaveToOmModel(model, file_name_prefix, model_buff);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][SaveToOmModel] Save to om model failed");
    return ret;
  }
  return SUCCESS;
}

Status GeGenerator::Impl::SaveRootModel(const std::string &file_name_prefix, const GeRootModelPtr &ge_root_model,
                                        ModelBufferData &model_buff, OfflineModelFormat om_format) const {
  bool is_unknown_shape = false;
  GE_ASSERT_SUCCESS(ge_root_model->CheckIsUnknownShape(is_unknown_shape),
                    "root model(id:%u) CheckIsUnknownShape failed", ge_root_model->GetModelId());
  GELOGD("begin save root model, cur model is %s", (is_unknown_shape ? "unknown shape model" : "known shape model"));
  GE_CHK_BOOL_EXEC(!ge_root_model->GetSubgraphInstanceNameToModel().empty(),
                   REPORT_INNER_ERR_MSG("E19999", "root model(id:%u) has no sub model.", ge_root_model->GetModelId());
                   return FAILED, "[Get][SubModel] ge root model has no sub model");
  GeModelPtr model_root = nullptr;
  if (is_unknown_shape) {
    auto name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
    model_root = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];
  } else {
    model_root = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  }
  GE_CHECK_NOTNULL(model_root);
  // set atc version
  if (!SetAtcVersionInfo(*(model_root.get()))) {
    GELOGW("SetPackageVersionInfo of atc failed!");
  }
  SetOmSystemInfo(*model_root.get());
  SetHcomGroupRanks(*(model_root.get()));
  GE_ASSERT_SUCCESS(ge_root_model->CheckAndSetNeedSoInOM(), "Check so in om failed, model id:%u.",
                    ge_root_model->GetModelId());
  SetHostEnvOsCpuInfo(ge_root_model, *(model_root.get()));
  if (IsMobile()) {
    GELOGI("[Mobile] set om_format to OM_FORMAT_MOBILE.");
    om_format = OfflineModelFormat::OM_FORMAT_MOBILE;
  }
  const auto model_save_helper = ModelSaveHelperFactory::Instance().Create(om_format);
  GE_CHECK_NOTNULL(model_save_helper);
  model_save_helper->SetSaveMode(is_offline_);

  // Configure attribute compression mode from options
  std::map<std::string, std::string> options;
  std::string attr_compression_value;
  if (GetContext().GetOption(ENABLE_ATTR_COMPRESSION, attr_compression_value) == SUCCESS) {
    (void)model_save_helper->ConfigureAttrCompressionMode(attr_compression_value);
  }

  GE_ASSERT_SUCCESS(model_save_helper->SaveToOmRootModel(ge_root_model, file_name_prefix, model_buff, is_unknown_shape),
                    "SaveToOmRootModel failed, model id:%u, om_format:%d", ge_root_model->GetModelId(), om_format);
  return SUCCESS;
}

Status GeGenerator::Impl::BuildModel(const Graph &graph, const std::vector<GeTensor> &inputs,
                                     GeRootModelPtr &ge_root_model) {
  static std::atomic<GraphId> atomic_graph_id(0);
  auto graph_id = atomic_graph_id.fetch_add(1);
  const std::map<std::string, std::string> options = GetThreadLocalContext().GetAllGraphOptions();
  Status ret = graph_manager_.AddGraph(graph_id, graph, options, omg_context_);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "add graph(id:%u) failed, ret:%u", graph_id, ret);
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "[Add][Graph] fail, graph id: %u", graph_id);
    (void)graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  graph_manager_.SetOptionsRunGraphFlag(false);
  return BuildModelWithGraphId(graph_id, inputs, ge_root_model, options);
}

Status GeGenerator::Impl::BuildModelWithGraphId(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                                                GeRootModelPtr &ge_root_model,
                                                const std::map<std::string, std::string> &options) {
  // if session_id_ is default value, current session_id need to be generated internal
  SessionId session_id = (session_id_ == UINT64_MAX) ? SessionIdManager::GetNextSessionId() : session_id_;
  // This is a temporary add for graph with variable
  auto version = static_cast<int32_t>(SessionVersion::ClOUD_VERSION);
  const auto manager = VarManager::Instance(session_id);
  GE_CHECK_NOTNULL(manager);
  Status ret = manager->Init(version, session_id, kDefaultDeviceId, kDefaultJobId);
  if (ret != SUCCESS) {
    GELOGW("Failed init var instance, session_id %lu", session_id);
  }
  ret = manager->SetAllMemoryMaxValue(options);
  GELOGI("Start init var instance, session_id %lu", session_id);
  if (ret != SUCCESS) {
    GELOGW("Failed SetAllMemoryMaxValue, session_id %lu", session_id);
  }
  if (is_singleop_unregistered_) {
    ret = graph_manager_.BuildGraphForUnregisteredOp(graph_id, inputs, ge_root_model, session_id);
  } else {
    ret = graph_manager_.BuildGraph(graph_id, inputs, ge_root_model, session_id);
  }

  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "build graph failed, graph id:%u, ret:%u", graph_id, ret);
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED, "[Build][Graph] fail, graph id: %u", graph_id);
  }
  if (session_id_ == UINT64_MAX) { // default session_id_ need to destroy session resource internal
    RtContextUtil::GetInstance().DestroyRtContexts(session_id);
    Analyzer::GetInstance()->DestroySessionJsonObject(session_id);
    VarManagerPool::Instance().RemoveVarManager(session_id);
  }
  return ret;
}

Status GeGenerator::Impl::GenerateInfershapeGraph(const Graph &graph) {
  static std::atomic<GraphId> atomic_graph_id(0);
  auto graph_id = atomic_graph_id.fetch_add(1);
  const std::map<std::string, std::string> options;
  Status ret = graph_manager_.AddGraph(graph_id, graph, options, omg_context_);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "add graph failed, graph id:%u, ret:%u", graph_id, ret);
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED, "[Add][Graph] failed, graph id: %u", graph_id);
    (void)graph_manager_.Finalize();
    return GE_GENERATOR_GRAPH_MANAGER_ADD_GRAPH_FAILED;
  }

  ret = graph_manager_.GenerateInfershapeGraph(graph_id);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "GenerateInfershapeGraph failed, graph id:%u, ret:%u", graph_id, ret);
    GELOGE(GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED,
           "[Generate][Graph] failed, graph id:%u, ret:%u", graph_id, ret);
    return GE_GENERATOR_GRAPH_MANAGER_BUILD_GRAPH_FAILED;
  }

  return SUCCESS;
}
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/ge_ir_build.h"

#include <vector>
#include "graph/utils/graph_utils_ex.h"
#include "common/helper/file_saver.h"
#include "common/model/ge_model.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/screen_printer.h"
#include "ge/ge_api_types.h"
#include "register/register_types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/ge_types.h"
#include "framework/generator/ge_generator.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/compute_graph.h"
#include "graph/ge_global_options.h"
#include "graph/ge_tensor.h"
#include "graph/opsproto_manager.h"
#include "base/registry/opp_package_utils.h"
#include "register/op_lib_register_impl.h"
#include "graph/passes/control_flow_and_stream/data_pass.h"
#include "graph/passes/feature/net_output_pass.h"
#include "graph/shape_refiner.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/passes/pass_manager.h"
#include "api/gelib/gelib.h"
#include "api/aclgrph/attr_options/attr_options.h"
#include "api/aclgrph/option_utils.h"
#include "common/single_op_parser.h"
#include "framework/common/helper/model_helper.h"
#include "graph/utils/op_type_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/option_supportion_checker/option_supportion_checker.h"
#include "graph/manager/session_id_manager.h"
#include "graph/manager/util/rt_context_util.h"
#include "analyzer/analyzer.h"
#include "graph/utils/type_utils.h"
#include "register/custom_pass_helper.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"

namespace ge {
namespace {
const std::string IR_OPTION_TARGET = "target";
const std::string IR_OPTION_MODE = "mode";
const std::string IR_OP_CONF_DELIMITER = ":";
const std::string IR_OPTION_LOG_LEVEL_DEFAULT = "default";
const std::string IR_OPTION_BUFFER_OPTIMIZE_DEFAULT = "l2_optimize";
const std::string IR_OPTION_DISABLE_DEFAULT = "0";
const std::string IR_OPTION_ENABLE_DEFAULT = "1";
const std::string IR_OPTION_ENABLE_COMPRESS_WEIGHT_DEFAULT = "false";
const std::string IR_OPTION_SPARSITY_DEFAULT = "0";
const std::string KEEP_DTYPE_OPTION = "keep_dtype";
const std::string kInputFormat = "input_format";
const std::string kShapeGeneralized = "shape_generalized";
const std::string kShapePrecise = "shape_precise";
const std::string kOffline = "offline";
/**
 * @name  SetOpAttrFun
 * @brief set attribute for operators in the configuration file
 * @param graph      [IN/OUT] compute graph
 * @param cfg_path   [IN] the config file path
 * @return graphStatus
 */
using SetOpAttrFun = graphStatus (*)(const ComputeGraphPtr &graph, const std::string &cfg_path);

const std::map<aclgrphAttrType, SetOpAttrFun> kAttrTypeFuncMap = {
    {ATTR_TYPE_KEEP_DTYPE, KeepDtypeFunc},
    {ATTR_TYPE_WEIGHT_COMPRESS, WeightCompressFunc}
};

const std::map<aclgrphAttrType, std::string> kAttrTypeToStringMap = {
    {ATTR_TYPE_KEEP_DTYPE, KEEP_DTYPE_OPTION},
    {ATTR_TYPE_WEIGHT_COMPRESS, ge::ir_option::COMPRESS_WEIGHT_CONF}
};

// ge ir场景，将jit_compile默认值设置为1
void SetJitCompileTrue(std::map<std::string, std::string> &options) {
  if (options.find(JIT_COMPILE) != options.cend()) {
    GELOGI("jit_compile option exists, value: %s", options[JIT_COMPILE].c_str());
    return;
  }
  GELOGI("jit_compile option does not exist, use default value: \"1\"");
  options[JIT_COMPILE] = "1";
}

void SetBuildGraphModeOffline(std::map<std::string, std::string> &options) {
  GELOGI("build graph mode option set value to offset");
  options[OPTION_BUILD_GRAPH_MODE] = kOffline;
}

// input_hint_shape暂不支持
Status CheckInputHintShape(const std::map<std::string, std::string> &global_options) {
  auto iter = global_options.find(INPUT_HINT_SHAPE);
  if (iter != global_options.end() && !iter->second.empty()) {
    const std::string reason = "Option[input_hint_shape: " +
      iter->second + "] is not supported in ge_ir_build. Please do not set it.";
    REPORT_PREDEFINED_ERR_MSG("E10055", std::vector({"reason"}), std::vector({reason.c_str()}));
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] %s", reason.c_str());
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

Status VerifyVarOffset(const ComputeGraphPtr &root_graph,
                       std::map<std::string, std::pair<int64_t, GeTensorDesc>> &var_name_to_verify_info) {
  GE_ASSERT_NOTNULL(root_graph);
  for (auto node : root_graph->GetAllNodesPtr()) {
    GE_ASSERT_NOTNULL(node);
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->GetType() != VARIABLE) {
      continue;
    }
    std::string placement;
    (void)AttrUtils::GetStr(node->GetOpDesc(), ATTR_VARIABLE_PLACEMENT, placement);
    GE_ASSERT_TRUE(placement != "host", "Bundle model only supports variables in device memory.");
    GE_ASSERT_TRUE(!op_desc->GetOutputOffset().empty(), "Variable [%s] has not been correctly compiled.",
                   op_desc->GetNamePtr());
    auto iter = var_name_to_verify_info.find(op_desc->GetName());
    if (iter == var_name_to_verify_info.end()) {
      var_name_to_verify_info[op_desc->GetName()] = {op_desc->GetOutputOffset()[0UL], op_desc->GetOutputDesc(0U)};
      return SUCCESS;
    }
    if (iter->second.first != op_desc->GetOutputOffset()[0UL]) {
      GELOGE(FAILED,
             "Models shared the same variable [%s] but have different offsets, please ensure that the models within "
             "the bundle model are compiled within the same session.",
             op_desc->GetNamePtr());
      return FAILED;
    }
    const GeTensorDesc &lhs = iter->second.second;
    const GeTensorDesc &rhs = op_desc->GetOutputDesc(0U);
    const bool equal_flag = (lhs.GetShape() == rhs.GetShape()) && (lhs.GetDataType() == rhs.GetDataType());
    if (!equal_flag) {
      GELOGE(FAILED,
             "Models shared the same variable [%s] but have different tensor_descs, please ensure that the models "
             "within the bundle model are compiled within the same session.",
             op_desc->GetNamePtr());
      return FAILED;
    }
  }

  return SUCCESS;
}

static graphStatus CheckGlobalOptions(std::map<std::string, std::string> &global_options) {
  // check param disable_reuse_memory
  std::string disable_reuse_memory = global_options.find(ge::ir_option::EXEC_DISABLE_REUSED_MEMORY) ==
                                         global_options.end()
                                         ? IR_OPTION_DISABLE_DEFAULT
                                         : global_options[ge::ir_option::EXEC_DISABLE_REUSED_MEMORY];
  GE_CHK_BOOL_EXEC(ge::CheckDisableReuseMemoryParamValid(disable_reuse_memory) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "[Check][DisableReuseMemory] failed!");
  global_options[ge::ir_option::EXEC_DISABLE_REUSED_MEMORY] = disable_reuse_memory;
  // check buffer_optimize
  std::string buffer_optimize = global_options.find(ge::ir_option::BUFFER_OPTIMIZE) == global_options.end()
                                    ? IR_OPTION_BUFFER_OPTIMIZE_DEFAULT
                                    : global_options[ge::ir_option::BUFFER_OPTIMIZE];
  GE_CHK_BOOL_EXEC(ge::CheckBufferOptimizeParamValid(buffer_optimize) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "[Check][BufferOptimize] failed!");
  global_options[ge::ir_option::BUFFER_OPTIMIZE] = buffer_optimize;
  // check enable_single_stream
  std::string enable_single_stream = global_options.find(ge::ir_option::ENABLE_SINGLE_STREAM) == global_options.end()
                                         ? ""
                                         : global_options[ge::ir_option::ENABLE_SINGLE_STREAM];
  GE_CHK_BOOL_EXEC(ge::CheckEnableSingleStreamParamValid(enable_single_stream) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "[Check][EnableSingleStream] failed!");

  // check external_weight
  std::string enable_external_weight = global_options.find(ge::ir_option::EXTERNAL_WEIGHT) == global_options.end()
                                           ? ""
                                           : global_options[ge::ir_option::EXTERNAL_WEIGHT];
  GE_CHK_BOOL_EXEC(ge::CheckExternalWeightParamValid(enable_external_weight) == ge::SUCCESS,
                   return ge::GRAPH_PARAM_INVALID, "[Check][ExternalWeight] failed!");

  // check ac_parallel_enable
  std::string ac_parallel_enable = global_options.find(ge::ir_option::AC_PARALLEL_ENABLE) == global_options.end()
                                       ? ""
                                       : global_options[ge::ir_option::AC_PARALLEL_ENABLE];
  GE_CHK_BOOL_EXEC(ge::CheckAcParallelEnableParamValid(ac_parallel_enable) == ge::SUCCESS,
                   return ge::GRAPH_PARAM_INVALID, "[Check][AcParallelEnable] failed!");
  // check tiling_schedule_optimize
  std::string tiling_schedule_optimize = global_options.find(ge::ir_option::TILING_SCHEDULE_OPTIMIZE) ==
                                             global_options.end()
                                             ? ""
                                             : global_options[ge::ir_option::TILING_SCHEDULE_OPTIMIZE];
  GE_CHK_BOOL_EXEC(ge::CheckTilingScheduleOptimizeParamValid(tiling_schedule_optimize) == ge::SUCCESS,
                    return ge::GRAPH_PARAM_INVALID, "[Check][TilingScheduleOptimize] failed!");

  // check quant_dumpable
  std::string quant_dumpable = global_options.find(ge::ir_option::QUANT_DUMPABLE) == global_options.end()
                                   ? ""
                                   : global_options[ge::ir_option::QUANT_DUMPABLE];
  GE_ASSERT_SUCCESS(CheckQuantDumpableParamValid(quant_dumpable), "[Check][QuantDumpable] failed!");

  // check compress_weight
  std::string enable_compress_weight = global_options.find(ge::ir_option::ENABLE_COMPRESS_WEIGHT) ==
                                           global_options.end()
                                           ? IR_OPTION_ENABLE_COMPRESS_WEIGHT_DEFAULT
                                           : global_options[ge::ir_option::ENABLE_COMPRESS_WEIGHT];
  std::string compress_weight_conf = global_options.find(ge::ir_option::COMPRESS_WEIGHT_CONF) == global_options.cend()
                                         ? ""
                                         : global_options[ge::ir_option::COMPRESS_WEIGHT_CONF];
  GE_CHK_BOOL_EXEC(ge::CheckCompressWeightParamValid(enable_compress_weight, compress_weight_conf) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "[Check][CompressWeight] failed!");
  global_options[ge::ir_option::ENABLE_COMPRESS_WEIGHT] = (enable_compress_weight == "true") ?
                                                     ge::kEnableCompressWeightTrue :
                                                     ge::kEnableCompressWeightFalse;
  // check sparsity option
  if (global_options.find(ir_option::SPARSITY) == global_options.end()) {
    global_options[ir_option::SPARSITY] = IR_OPTION_SPARSITY_DEFAULT;
  }
  GE_CHK_BOOL_EXEC(CheckSparseParamValid(global_options[ir_option::SPARSITY]) == SUCCESS,
      return GRAPH_PARAM_INVALID, "[Check][Sparsity] failed!");
  // check optypelist_for_implmode and op_select_implmode
  std::string optypelist_for_implmode = global_options.find(ge::ir_option::OPTYPELIST_FOR_IMPLMODE) ==
                                            global_options.end()
                                            ? ""
                                            : global_options[ge::ir_option::OPTYPELIST_FOR_IMPLMODE];
  std::string op_select_implmode = global_options.find(ge::ir_option::OP_SELECT_IMPL_MODE) ==
                                       global_options.cend()
                                       ? ""
                                       : global_options[ge::ir_option::OP_SELECT_IMPL_MODE];
  GE_CHK_BOOL_EXEC(
      ge::CheckImplmodeParamValid(optypelist_for_implmode, op_select_implmode) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "[Check][Implmode] failed!");
  global_options[ge::ir_option::OP_SELECT_IMPL_MODE] = op_select_implmode;

  // set precision mode default value
  const std::string precision_mode = global_options.find(ge::ir_option::PRECISION_MODE) ==
                                     global_options.end()
                                     ? ""
                                     : global_options[ge::ir_option::PRECISION_MODE];
  const std::string precision_mode_v2 = global_options.find(ge::ir_option::PRECISION_MODE_V2) ==
                                        global_options.end()
                                        ? ""
                                        : global_options[ge::ir_option::PRECISION_MODE_V2];

  // check allow_hf32
  GE_ASSERT_SUCCESS(CheckAllowHF32ParamValid(global_options[ir_option::ALLOW_HF32]),
                    "[Check][AllowHF32]failed!");
  // check modify_mixlist
  std::string modify_mixlist = global_options.find(ge::ir_option::MODIFY_MIXLIST) ==
                               global_options.cend()
                               ? ""
                               : global_options[ge::ir_option::MODIFY_MIXLIST];

  GE_ASSERT_SUCCESS(CheckPrecisionModeParamValid(precision_mode), "[Check][PrecisionMode]failed!");
  GE_ASSERT_SUCCESS(CheckPrecisionModeV2ParamValid(precision_mode_v2), "[Check][PrecisionModeV2]failed!");
  GE_ASSERT_SUCCESS(CheckPrecisionModeV2Conflict(precision_mode, precision_mode_v2),
                    "[Check][PrecisionModeV2Conflict]failed!");
  if (ge::CheckModifyMixlistParamValid(precision_mode, precision_mode_v2, modify_mixlist) !=
      ge::SUCCESS) {
    return ge::GRAPH_PARAM_INVALID;
  }
  global_options[ge::ir_option::MODIFY_MIXLIST] = modify_mixlist;
  global_options[ge::OPTION_EXEC_HCCL_FLAG] = IR_OPTION_ENABLE_DEFAULT;
  if (CheckHostEnvOsAndHostEnvCpuValid(global_options[OPTION_HOST_ENV_OS], global_options[OPTION_HOST_ENV_CPU])
      != SUCCESS) {
    return GRAPH_PARAM_INVALID;
  }
  GE_ASSERT_SUCCESS(CheckScreenPrinterOption(global_options), "[Check][ge.screen_print_mode]failed!");
  GE_ASSERT_GRAPH_SUCCESS(CheckOptimizationOptionValid(global_options));
  return GRAPH_SUCCESS;
}

static void LoadOpsProto() {
  std::string opsproto_path;
  Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  if (ret != SUCCESS) {
    GELOGW("Unsuccessful to get ops proto path!");
  }
  GELOGI("Get opsproto path is %s", opsproto_path.c_str());
  OpsProtoManager *manager = OpsProtoManager::Instance();
  std::map<std::string, std::string> option_tmp;
  option_tmp.emplace(std::pair<std::string, std::string>(string("ge.opsProtoLibPath"), opsproto_path));
  (void)manager->Initialize(option_tmp);
  gert::OppPackageUtils::LoadAllOppPackage();
}

static graphStatus aclgrphBuildInitializeImpl(std::map<std::string, std::string> &global_options) {
  GELOGD("Enter aclgrphInitialize start!");
  SetDefaultHostEnvOsAndHostEnvCpu(global_options[OPTION_HOST_ENV_OS], global_options[OPTION_HOST_ENV_CPU]);
  SetJitCompileTrue(global_options);
  SetBuildGraphModeOffline(global_options);
  // check supported global options
  if (IrbuildCheckSupportedGlobalOptions(global_options) != GRAPH_SUCCESS) {
    GELOGW("[Check][Supported Global Options] unsuccessful!");
  }

  // check global options
  if (CheckGlobalOptions(global_options) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Global Options] falied!");
    return GRAPH_PARAM_INVALID;
  }
  ScreenPrinter::GetInstance().Init(global_options[OPTION_SCREEN_PRINT_MODE]);

  auto iter = global_options.find("ge.autoTuneMode");
  if (iter != global_options.end() && !iter->second.empty()) {
    const std::string reason =
        "The configured value is not supported. The Auto Tune function has been deprecated. "
        "Please use the AOE tool for tuning";
    REPORT_PREDEFINED_ERR_MSG("E10055", std::vector<const char_t *>({"reason"}),
                              std::vector<const char_t *>({reason.c_str()}));
    GELOGE(GRAPH_FAILED,
           "[Check][Param]Options unsupport, The Auto Tune function has been discarded. Please use the AOE tool for "
           "tuning.");
    return GRAPH_FAILED;
  }
  GE_ASSERT_GRAPH_SUCCESS(CheckInputHintShape(global_options));
  // print global option map
  ge::PrintOptionMap(global_options, "global option");
  GE_ASSERT_GRAPH_SUCCESS(OpLibRegistry::GetInstance().PreProcessForCustomOp());
  LoadOpsProto();
  GE_ASSERT_SUCCESS(CustomPassHelper::Instance().Load());

  std::shared_ptr<ge::GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGI("aclgrphInitialize start!");
    auto ret = ge::GELib::Initialize(global_options);
    if (ret != ge::SUCCESS) {
      GELOGE(ret, "[Init][GELib] failed!");
      return GRAPH_FAILED;
    }
  }
  GELOGW("gelib has been initialized!");
  Status ret = static_cast<uint32_t>(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE));
  GE_ASSERT_SUCCESS(ret, "ErrorManager init failed!");
  return GRAPH_SUCCESS;
}

graphStatus aclgrphBuildInitialize(std::map<std::string, std::string> global_options) {
  return aclgrphBuildInitializeImpl(global_options);
}

graphStatus aclgrphBuildInitialize(std::map<AscendString, AscendString> &global_options) {
  std::map<std::string, std::string> tmp_global_options;
  for (auto &option : global_options) {
    if (option.first.GetString() == nullptr || option.second.GetString() == nullptr) {
      GELOGE(GRAPH_FAILED, "[Check][Options]AclgrphBuildInitialize option is nullptr.");
      return GRAPH_FAILED;
    }
    std::string key = option.first.GetString();
    std::string val = option.second.GetString();
    tmp_global_options[key] = val;
  }
  return aclgrphBuildInitializeImpl(tmp_global_options);
}

void aclgrphBuildFinalize() {
  CustomPassHelper::Instance().Unload();
  if (ge::GELib::GetInstance() != nullptr && ge::GELib::GetInstance()->InitFlag()) {
    (void)ge::GELib::GetInstance()->Finalize();
    return;
  }
  GELOGW("[Notice] gelib has not been initialized!do nothing!");
}

class Impl {
 public:
  Impl() {
    omg_context_ = domi::GetContext();
    omg_context_.format = domi::DOMI_TENSOR_ND;
    omg_context_.input_nodes_format_map.clear();
    omg_context_.output_formats.clear();
    omg_context_.user_input_dims.clear();
    omg_context_.input_dims.clear();
    omg_context_.op_conf_map.clear();
    omg_context_.out_nodes_map.clear();
    omg_context_.user_out_nodes.clear();
    omg_context_.net_format = domi::DOMI_TENSOR_RESERVED;
    omg_context_.type = domi::FRAMEWORK_RESERVED;
    omg_context_.run_mode = RunMode::ONLY_PRE_CHECK;
    omg_context_.train_flag = false;
    omg_context_.output_type.clear();
    omg_context_.is_dynamic_input = false;
    omg_context_.dynamic_batch_size.clear();
    omg_context_.dynamic_image_size.clear();
    omg_context_.dynamic_dims.clear();
    omg_context_.user_attr_index_valid = false;
  };
  ~Impl() { (void)generator_.Finalize(); };
  graphStatus CheckBuildModeAndBuildStep();
  graphStatus GetSupportedOptions(const std::map<std::string, std::string> &in,
                                  std::map<std::string, std::string> &out) const;
  graphStatus CheckOptions(const std::map<std::string, std::string> &options);
  graphStatus CreateInputsForIRBuild(const ge::Graph &graph, std::vector<ge::GeTensor> &inputs);
  graphStatus SetInputs(std::vector<ge::GeTensor> &inputs, const std::vector<ge::NodePtr> &data_nodes);
  graphStatus UpdateDataOpAttr(const Graph &graph,
                               const std::string &input_shape,
                               const std::string &input_shape_range,
                               const std::string &input_format);
  graphStatus UpdateDataOpAttr(const Graph &graph);
  graphStatus CheckDataOpAttrIndexValid(const Graph &graph, const std::string &input_shape_range);
  graphStatus Init(const Graph &graph, const std::map<std::string, std::string> &options);
  graphStatus CheckAutoTuneMode(const std::map<std::string, std::string> &options) const;
  graphStatus SplitForVariableInferGraph(const ComputeGraphPtr &origin_graph,
                                         const std::vector<std::string> &const_names,
                                         WeightRefreshableGraphs &weight_refreshable_graphs) const;
  graphStatus GenerateVariableInferGraph(const ComputeGraphPtr &origin_graph,
                                          const vector<std::string> &const_names, Graph &infer_graph,
                                          std::vector<ge::NodePtr> &const_nodes,
                                          std::vector<ge::NodePtr> &var_nodes) const;
  graphStatus GenerateVariableInitGraph(const std::vector<ge::NodePtr> &var_nodes,
                                        const std::vector<ge::NodePtr> &const_nodes, Graph &init_graph) const;
  graphStatus GenerateVariableUpdateGraph(const std::vector<ge::NodePtr> &var_nodes, Graph &update_graph) const;

  NodePtr InsertOp(const ComputeGraphPtr &compute_graph, const std::string &node_type,
                   const std::string &node_name,
                   const std::vector<GeTensorDesc> &input_list,
                   const std::vector<GeTensorDesc> &output_list) const;

  NodePtr InsertIfNode(const ge::NodePtr &var_node, ComputeGraphPtr &compute_graph) const;

  graphStatus ConstructIfSubgraphs(const ge::NodePtr &var_node, ge::NodePtr &if_node,
                                   ComputeGraphPtr &compute_graph) const;

  graphStatus BuildModel(const Graph &graph, const std::map<std::string, std::string> &options,
                         ModelBufferData &model);
  graphStatus InitDomiOmgContext(const std::string &input_shape, const std::string &input_format,
                                 bool is_dynamic_input);
  graphStatus GetInputShapeRange(const std::string &input_shape_range,
                                 std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> &name_shape_range_map,
                                 std::vector<std::vector<std::pair<int64_t, int64_t>>> &index_shape_range_map) const;
  static graphStatus InferShapePrepare(const ComputeGraphPtr &compute_graph);
  bool GetUsrAttrIndexValidFlag();
  bool IsAttrIndexSetByUser(const ComputeGraphPtr &compute_graph, size_t &data_num,
                            std::vector<int64_t> &attr_index) const;
  void SetRtSocVersion() const;
  void UpdateThreadContext();
  void LoadOpsProto();
  std::string GetParam(const std::string &param);
 public:
  ge::GeGenerator generator_;
  std::map<std::string, std::string> options_;
  bool is_dynamic_input_ = false;
  OmgContext omg_context_;
  uint64_t session_id_ = UINT64_MAX;
  std::shared_ptr<GraphRebuildStateCtrl> rebuild_state_ctrl_;
};

graphStatus Impl::InferShapePrepare(const ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);

  PassManager prepare_infershape;
  prepare_infershape.AddPass("PrepareNetoutput", new(std::nothrow) NetOutputPass);
  prepare_infershape.AddPass("PrepareSubGraphReflection", new (std::nothrow) DataPass);

  auto ret = prepare_infershape.Run(compute_graph);
  if ((ret != SUCCESS) && (ret != NOT_CHANGED)) {
    GELOGE(ret, "[Prepair][InferShape] failed, ret:%d", ret);
    return ret;
  }
  GELOGD("Prepair for infershape success!");
  return GRAPH_SUCCESS;
}

bool Impl::GetUsrAttrIndexValidFlag() {
  return omg_context_.user_attr_index_valid;
}

bool Impl::IsAttrIndexSetByUser(const ComputeGraphPtr &compute_graph,
                                size_t &data_num,
                                std::vector<int64_t> &attr_index) const {
  bool all_zero_flag = true;
  for (ge::NodePtr &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    ge::OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (OpTypeUtils::IsDataNode(op->GetType())) {
      data_num++;
      int64_t index = 0;
      if (AttrUtils::GetInt(op, ATTR_NAME_INDEX, index)) {
        if (index != 0) {
          all_zero_flag = false;
        }
        attr_index.push_back(index);
      } else {
        GELOGW("[Get][AttrIndex] Get index[%ld] unsuccessful for op[%s].", index, op->GetName().c_str());
      }
    }
  }
  if (data_num > 1 && attr_index.size() == data_num && all_zero_flag) {
    GELOGI("Attr indexes are not set by user.");
    return false;
  }
  return true;
}

graphStatus Impl::GetInputShapeRange(
    const std::string &input_shape_range,
    std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> &name_shape_range_map,
    std::vector<std::vector<std::pair<int64_t, int64_t>>> &index_shape_range_map) const {
  if (input_shape_range.empty()) {
    GELOGI("Input shape range is empty.");
    return GRAPH_SUCCESS;
  }
  Status ret = GRAPH_PARAM_INVALID;
  if (input_shape_range.find(":") != std::string::npos) {
    ret = ParseInputShapeRange(input_shape_range, name_shape_range_map);
  } else {
    ret = ParseInputShapeRange(input_shape_range, index_shape_range_map);
  }
  if (ret != SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "[Parse][InputShapeRange] parse shape range[%s] failed.", input_shape_range.c_str());
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

graphStatus Impl::CheckDataOpAttrIndexValid(const Graph &graph, const std::string &input_shape_range) {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  // when set input shape range by index, user must set data attr index, eg. "[1, 3, 3, -1],[1, 3~5, 6, -1]"
  bool index_input_shape_range_flag = !input_shape_range.empty() && (input_shape_range.find(":") == std::string::npos);
  size_t data_num = 0;
  std::vector<int64_t> attr_index;
  if (!IsAttrIndexSetByUser(compute_graph, data_num, attr_index)) {
    if (index_input_shape_range_flag) {
      std::string situation = "Data op index";
      std::string reason = "When setting the input shape range by index, you must set the index attribute of all DATA operators";
      REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"situation", "reason"}),
                                std::vector<const char_t *>({situation.c_str(), reason.c_str()}));
      GELOGE(GRAPH_FAILED, "[Check][AttrIndex] Data op index is not set, total data op num[%ld], "
             "when set input shape range by index.", data_num);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  omg_context_.user_attr_index_valid = true;
  for (size_t i = 0; i < data_num; ++i) {
    if (std::find(attr_index.begin(), attr_index.end(), i) == attr_index.end()) {
      omg_context_.user_attr_index_valid = false;
      if (index_input_shape_range_flag) {
        std::string situation = "Data op index[" + std::to_string(i) + "]";
        std::string reason = "When setting the input shape range by index, you must set the index attribute of all DATA operators";
        REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"situation", "reason"}),
                                  std::vector<const char_t *>({situation.c_str(), reason.c_str()}));
        GELOGE(GRAPH_FAILED, "[Check][AttrIndex] Attr index [%ld] is not set, total data op num[%ld], "
               "when set input shape range by index", i, data_num);
        return GRAPH_FAILED;
      } else {
        GELOGW("[Check][AttrIndex] Attr index [%ld] is not set, total data op num[%ld].", i, data_num);
      }
    }
  }
  GELOGI("Data op attr indexes are set by user and valid.");
  return GRAPH_SUCCESS;
}

graphStatus Impl::UpdateDataOpAttr(const Graph &graph) {
  std::string input_shape = GetParam(ge::ir_option::INPUT_SHAPE);
  std::string input_format = GetParam(ge::ir_option::INPUT_FORMAT);
  std::string input_shape_range = GetParam(ge::INPUT_SHAPE_RANGE);
  std::string dynamic_batch_size = GetParam(ge::ir_option::DYNAMIC_BATCH_SIZE);
  std::string dynamic_image_size = GetParam(ge::ir_option::DYNAMIC_IMAGE_SIZE);
  std::string dynamic_dims = GetParam(ge::ir_option::DYNAMIC_DIMS);
  if (CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
      dynamic_batch_size, dynamic_image_size, dynamic_dims) != SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][TransferShapeAndRange] failed!");
    return GRAPH_PARAM_INVALID;
  }

  return UpdateDataOpAttr(graph, input_shape, input_shape_range, input_format);
}

graphStatus Impl::UpdateDataOpAttr(const Graph &graph,
                                   const std::string &input_shape,
                                   const std::string &input_shape_range,
                                   const std::string &input_format) {
  GELOGD("Enter Update Data Attr Process!");
  graphStatus ret = CheckDataOpAttrIndexValid(graph, input_shape_range);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Check][DataOpAttrIndex] fail, shape range[%s].", input_shape_range.c_str());
    return GRAPH_FAILED;
  }
  std::map<std::string, std::vector<int64_t>> shape_map;
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_shape_map;
  if (!input_shape.empty()) {
    GE_CHK_BOOL_EXEC(ParseInputShape(input_shape, shape_map, user_shape_map, true),
                     return GRAPH_PARAM_INVALID, "[Parse][InputShape] failed!");
  }
  std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> name_shape_range_map;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> index_shape_range_map;
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  ret = GetInputShapeRange(input_shape_range, name_shape_range_map, index_shape_range_map);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Get][InputShapeRange] fail, shape range[%s].", input_shape_range.c_str());
    return GRAPH_FAILED;
  }
  for (ge::NodePtr &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    ge::OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (OpTypeUtils::IsDataNode(op->GetType())) {
      if (UpdateDataOpShape(op, shape_map) != SUCCESS) {
        GELOGE(GRAPH_FAILED, "[Update][DataOpShape] fail for op:%s.", op->GetName().c_str());
        return GRAPH_FAILED;
      }
      if (UpdateDataOpShapeRange(op, name_shape_range_map) != SUCCESS) {
        GELOGE(GRAPH_FAILED, "[Update][DataOpShapeRange] fail for op:%s.", op->GetName().c_str());
        return GRAPH_FAILED;
      }
      if (UpdateDataOpShapeRange(op, index_shape_range_map) != SUCCESS) {
        GELOGE(GRAPH_FAILED, "[Update][DataOpShapeRange] fail for op:%s.", op->GetName().c_str());
        return GRAPH_FAILED;
      }
      UpdateDataOpFormat(op, input_format);
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus Impl::CheckBuildModeAndBuildStep() {
  std::string build_mode;
  auto it = options_.find(BUILD_MODE);
  if (it != options_.end() && !(it->second.empty())) {
    if (build_mode_options.find(it->second) == build_mode_options.end()) {
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
          std::vector<const char_t *>({BUILD_MODE, it->second.c_str(), "The current value is not within the valid range."}));
      GELOGE(GRAPH_PARAM_INVALID, "[Check][BuildMode]:%s is unsupported. Please check!", it->second.c_str());
      return GRAPH_PARAM_INVALID;
    }
    build_mode = it->second;
  }
  it = options_.find(BUILD_STEP);
  if (it != options_.end() && !(it->second.empty())) {
    if (build_step_options.find(it->second) == build_step_options.end()) {
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
          std::vector<const char_t *>({BUILD_STEP, it->second.c_str(), "The current value is not within the valid range."}));
      GELOGE(GRAPH_PARAM_INVALID, "[Check][BuildStep]:%s is unsupported. Please check!", it->second.c_str());
      return GRAPH_PARAM_INVALID;
    }
  } else {
    if (build_mode == BUILD_MODE_TUNING) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
          std::vector<const char_t *>(
              {BUILD_STEP, "null", "If the build mode is set to TUNING, the build step must be specified."}));
      GELOGE(GRAPH_PARAM_INVALID, "[Check][BUILD_STEP] tuning must specify build step. Please check!");
      return GRAPH_PARAM_INVALID;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus Impl::GetSupportedOptions(const std::map<std::string, std::string> &in,
                                      std::map<std::string, std::string> &out) const {
  for (auto &ele : in) {
    std::set<std::string>::const_iterator it = ge::ir_option::ir_builder_suppported_options.find(ele.first);
    if (it == ge::ir_option::ir_builder_suppported_options.cend()) {
      const std::set<std::string>::const_iterator it_lx_fusion =
          ir_builder_supported_options_for_lx_fusion.find(ele.first);
      if (it_lx_fusion == ir_builder_supported_options_for_lx_fusion.cend()) {
        std::set<std::string>::const_iterator it_inner = ge::ir_builder_suppported_options_inner.find(ele.first);
        if (it_inner == ge::ir_builder_suppported_options_inner.cend()) {
          GELOGE(GRAPH_PARAM_INVALID, "[Check][Options] unsupported option(%s), Please check!",
                 ele.first.c_str());
          return GRAPH_PARAM_INVALID;
        }
      }
    }
    out.insert(ele);
  }
  return GRAPH_SUCCESS;
}

graphStatus Impl::CheckOptions(const std::map<std::string, std::string> &options) {
  auto ret = GetSupportedOptions(options, options_);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  // Check options build_mode and build_step.
  ret = CheckBuildModeAndBuildStep();
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  // Check option EXEC_DISABLE_REUSED_MEMORY
  auto it = options_.find(ge::ir_option::EXEC_DISABLE_REUSED_MEMORY);
  if (it != options_.end() && (CheckDisableReuseMemoryParamValid(it->second) != GRAPH_SUCCESS)) {
    return GRAPH_PARAM_INVALID;
  }
  GE_ASSERT_SUCCESS(CheckPrecisionModeParamValid(options_));
  // Check option modify_mixlist
  if (ge::CheckModifyMixlistParamValid(options_) != GRAPH_SUCCESS) {
    return GRAPH_PARAM_INVALID;
  }
  // Check option OP_PRECISION_MODE
  it = options_.find(ge::ir_option::OP_PRECISION_MODE);
  if (it != options_.end() && !it->second.empty() && !ge::CheckInputPathValid(it->second)) {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
                              std::vector<const char_t *>({ge::ir_option::OP_PRECISION_MODE, it->second.c_str(),
                                                           "Path defined by op precision mode is not found."}));
    GELOGE(GRAPH_PARAM_INVALID, "[Check][OP_PRECISION_MODE] %s not found", it->second.c_str());
    return GRAPH_PARAM_INVALID;
  }
  if (it != options_.end()) {
    GELOGI("Option set successfully, option_key=%s, option_value=%s",
           ge::ir_option::OP_PRECISION_MODE, it->second.c_str());
  }
  // Check Input Format
  if (options_.find(kInputFormat) != options_.end()) {
    return CheckInputFormat(options_[kInputFormat]);
  }
  const std::map<std::string, std::string>::const_iterator build_mode_iter =
      options_.find(ge::ir_option::SHAPE_GENERALIZED_BUILD_MODE);
  if (build_mode_iter != options_.cend()) {
    GELOG_DEPRECATED(ge::ir_option::SHAPE_GENERALIZED_BUILD_MODE);
  }
  bool mode_is_invalid = build_mode_iter != options_.cend() &&
                         build_mode_iter->second != kShapeGeneralized && build_mode_iter->second != kShapePrecise;
  if (mode_is_invalid) {
    REPORT_INNER_ERR_MSG("E19999", "Value[%s] of SHAPE_GENERALIZED_BUILD_MODE is invalid, must be shape_generalized "
                                 "or shape_precise.", build_mode_iter->second.c_str());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][SHAPE_GENERALIZED_BUILD_MODE]Shape generalized build mode %s is invalid, "
                                "only support shape_generalized or shape_precise", build_mode_iter->second.c_str());
    return GRAPH_PARAM_INVALID;
  }
  GE_ASSERT_GRAPH_SUCCESS(CheckOptimizationOptionValid(options_));
  return GRAPH_SUCCESS;
}

std::string Impl::GetParam(const std::string &param) {
  return options_.find(param) == options_.end() ? "" : options_[param];
}

graphStatus Impl::Init(const Graph &graph, const std::map<std::string, std::string> &options) {
  // 1. check options
  graphStatus ret = CheckOptions(options);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Check][Options] options are illegal! Please check!");
    return ret;
  }
  std::string input_shape = GetParam(ge::ir_option::INPUT_SHAPE);
  std::string input_hint_shape = GetParam(ge::ir_option::INPUT_HINT_SHAPE);
  std::string input_format = GetParam(ge::ir_option::INPUT_FORMAT);
  std::string input_shape_range = GetParam(ge::INPUT_SHAPE_RANGE);
  std::string dynamic_batch_size = GetParam(ge::ir_option::DYNAMIC_BATCH_SIZE);
  std::string dynamic_image_size = GetParam(ge::ir_option::DYNAMIC_IMAGE_SIZE);
  std::string dynamic_dims = GetParam(ge::ir_option::DYNAMIC_DIMS);
  if (CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
      dynamic_batch_size, dynamic_image_size, dynamic_dims) != SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][TransferShapeAndRange] failed!");
    return GRAPH_PARAM_INVALID;
  }

  ret = UpdateDataOpAttr(graph, input_shape, input_shape_range, input_format);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  std::string build_mode = (options_.find(BUILD_MODE) == options_.cend() || options_[BUILD_MODE] == BUILD_MODE_NORMAL)
                           ? "" : options_[BUILD_MODE];
  options_[BUILD_MODE] = build_mode;
  // set log level
  std::string log = options_.find(ge::ir_option::LOG_LEVEL) == options_.end()
                        ? IR_OPTION_LOG_LEVEL_DEFAULT
                        : options_[ge::ir_option::LOG_LEVEL];
  GE_CHK_BOOL_RET_STATUS_NOLOG(ge::CheckLogParamValidAndSetLogLevel(log) == 0, GRAPH_PARAM_INVALID);
  options_[ge::ir_option::LOG_LEVEL] = log;

  auto ret_status = CheckHintShapeConflictWithDynamicParam(input_hint_shape, dynamic_batch_size,
                                                           dynamic_image_size, dynamic_dims);
  if (ret_status != ge::SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][inputHintShape] failed!");
    return GRAPH_PARAM_INVALID;
  }

  auto status = CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims, input_shape,
                                            input_shape_range, input_format, is_dynamic_input_);
  if (status != ge::SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][DynamicInput] failed!");
    return GRAPH_PARAM_INVALID;
  }
  GELOGD("User input dynamic_batch_size:%s, dynamic_image_size:%s, dynamic_dims:%s.", dynamic_batch_size.c_str(),
         dynamic_image_size.c_str(), dynamic_dims.c_str());
  omg_context_.dynamic_batch_size = dynamic_batch_size;
  omg_context_.dynamic_image_size = dynamic_image_size;
  omg_context_.dynamic_dims = dynamic_dims;

  // check output_type
  std::string output_type = GetParam(ge::ir_option::OUTPUT_TYPE);
  GE_CHK_BOOL_EXEC(ge::CheckOutputTypeParamValid(output_type) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "[Check][OutputType] failed!");

  // check insert_op_conf
  std::string insert_op_conf = GetParam(ge::ir_option::INSERT_OP_FILE);
  GE_CHK_BOOL_EXEC(ge::CheckInsertOpConfParamValid(std::string(insert_op_conf)) == ge::SUCCESS,
      return ge::GRAPH_PARAM_INVALID, "[Check][InsertOpConf] failed!");

  GE_CHK_BOOL_EXEC(insert_op_conf.empty() || dynamic_dims.empty(),
                   return ge::GRAPH_PARAM_INVALID, "[Check][Data]dynamic dims function does not support aipp");

  // for IR builder.Only support om mode, so here fixed;
  options_.insert(std::pair<std::string, std::string>(string(IR_OPTION_MODE), to_string(0)));
  options_.insert(std::pair<std::string, std::string>(string(IR_OPTION_TARGET), "mini"));
  options_.insert(std::pair<std::string, std::string>(string(ge::RUN_FLAG), to_string(0)));
  options_.insert(std::pair<std::string, std::string>(string(ge::TRAIN_FLAG), to_string(0)));
  options_.insert(std::pair<std::string, std::string>(string(ge::SAVE_ORIGINAL_MODEL), to_string(0)));
  options_.insert(std::pair<std::string, std::string>(string(ge::OPTION_GRAPH_RUN_MODE), to_string(0)));

  // ge ir场景，将jit_compile默认值设置为1
  SetJitCompileTrue(options_);

  SetBuildGraphModeOffline(options_);

  // print ge option map
  ge::PrintOptionMap(options_, "ge option");

  SetRtSocVersion();
  UpdateThreadContext();
  // 3. init generator with options_
  ret = generator_.Initialize(options_, omg_context_);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Init][Generator]failed!");
    return ret;
  }
  // 4.parse and init Context with input shape format and net format info
  ret = this->InitDomiOmgContext(input_shape, input_format, is_dynamic_input_);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Init][DomiOmgContext]failed!");
    return ret;
  }
  return GRAPH_SUCCESS;
}

void Impl::SetRtSocVersion() const {
  auto &global_options_mutex = GetGlobalOptionsMutex();
  const std::lock_guard<std::mutex> lock(global_options_mutex);
  const auto &global_options = GetMutableGlobalOptions();
  auto it = global_options.find(ge::SOC_VERSION);
  if (it != global_options.end()) {
    const char *soc_version = it->second.c_str();
    rtError_t rt_ret = rtSetSocVersion(soc_version);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Set soc version %s unsuccessful. ret:0x%X", soc_version, rt_ret);
    }
    GELOGD("Set soc version %s success.", soc_version);
  }
}

void Impl::UpdateThreadContext() {
  auto &global_options_mutex = GetGlobalOptionsMutex();
  const std::lock_guard<std::mutex> lock(global_options_mutex);
  GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  GetThreadLocalContext().SetGraphOption(options_);
}

graphStatus Impl::CreateInputsForIRBuild(const ge::Graph &graph, std::vector<ge::GeTensor> &inputs) {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  std::vector<ge::NodePtr> data_nodes;
  for (ge::NodePtr &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    if (OpTypeUtils::IsDataNode(input_node->GetType())) {
      data_nodes.push_back(input_node);
    }
  }
  inputs.resize(data_nodes.size());
  return SetInputs(inputs, data_nodes);
}

graphStatus Impl::SetInputs(std::vector<ge::GeTensor> &inputs, const std::vector<ge::NodePtr> &data_nodes) {
  int64_t index = 0;
  for (const ge::NodePtr &data_node : data_nodes) {
    const auto op_desc = data_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (!GetUsrAttrIndexValidFlag()) {
      (void)AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, index);
      ++index;
    }
    int64_t id_index;
    (void)AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, id_index);
    const auto tensor_desc = op_desc->GetInputDescPtr(0);
    GE_CHECK_NOTNULL(tensor_desc);
    const std::string &data_op_name = op_desc->GetName();
    ge::GeShape data_shape;
    auto iter = omg_context_.input_dims.find(data_op_name);
    if (iter != omg_context_.input_dims.end()) {
      data_shape = ge::GeShape(iter->second);
      GELOGD("Data op: %s get shape from Context.", data_op_name.c_str());
    } else {
      data_shape = tensor_desc->GetShape();
      GELOGD("Data op: %s get shape from InputDesc in ge ir graph.", data_op_name.c_str());
    }
    // If user point input format, do work for all data ops; else do according to tensor_desc
    const auto data_format = omg_context_.format != domi::DOMI_TENSOR_ND ?
                             ge::TypeUtilsInner::DomiFormatToFormat(omg_context_.format) : tensor_desc->GetFormat();
    const ge::DataType data_type = tensor_desc->GetDataType();
    GELOGD("Data op get data type:%s from InputDesc in ge ir graph.",
           ge::TypeUtils::DataTypeToSerialString(data_type).c_str());

    ge::GeTensorDesc desc = tensor_desc->Clone();
    desc.SetShape(data_shape);
    desc.SetFormat(ge::Format(data_format));
    desc.SetDataType(data_type);
    ge::GeTensor input_tensor;
    input_tensor.SetTensorDesc(desc);
    GE_ASSERT_TRUE(static_cast<size_t>(id_index) < inputs.size(),
      "id_index %ld should be smaller than inputs size %zu", id_index, inputs.size());
    inputs[id_index] = input_tensor;
  }
  GELOGD("CreateInputsForIRBuild, inputs size: %zu", inputs.size());
  return GRAPH_SUCCESS;
}

graphStatus Impl::CheckAutoTuneMode(const std::map<std::string, std::string> &options) const {
  auto iter = options.find("ge.autoTuneMode");
  if ((iter != options.end()) && (!iter->second.empty())) {
    const std::string reason =
        "The configured value is not supported. The Auto Tune function has been deprecated. "
        "Please use the AOE tool for tuning";
    REPORT_PREDEFINED_ERR_MSG("E10055", std::vector<const char_t *>({"reason"}),
                              std::vector<const char_t *>({reason.c_str()}));
    GELOGE(
        GRAPH_FAILED,
        "[Check][Param]Options[%s] unsupport, The Auto Tune function has been discarded. Please use the AOE tool for "
        "tuning.",
        iter->first.c_str());
    return GRAPH_FAILED;
  }
  return SUCCESS;
}

graphStatus Impl::BuildModel(const Graph &graph, const std::map<std::string, std::string> &options,
                             ModelBufferData &model) {
  graphStatus ret = CheckAutoTuneMode(options);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Check][option] AutoTune mode is not supported!");
    return ret;
  }
  GE_ASSERT_SUCCESS(CheckInputHintShape(options));
  ret = Init(graph, options);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Init][GeGenerator]Build ir model Init failed!");
    return ret;
  }
  if (session_id_ != UINT64_MAX) {
    generator_.SetCurrentSessionId(session_id_);
  }
  if (rebuild_state_ctrl_ != nullptr) {
    generator_.SetExternalGraphRebuildStateCtrl(rebuild_state_ctrl_.get());
  }
  // 2. construct input
  std::vector<GeTensor> inputs;
  if (!omg_context_.is_dynamic_input) {  // if dynamic input , no need to creat inputs
    ret = CreateInputsForIRBuild(graph, inputs);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(ret, "[Create][InputsForIRBuild] failed!");
      return ret;
    }
  }

  // 3. build IR model
  ret = generator_.GenerateOnlineModel(graph, inputs, model);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Generate][OnlineModel] failed!");
    return ret;
  }
  return GRAPH_SUCCESS;
}

graphStatus Impl::InitDomiOmgContext(const std::string &input_shape, const std::string &input_format,
                                     bool is_dynamic_input) {
  // Clear omgcontext data first
  omg_context_.input_dims.clear();
  omg_context_.user_input_dims.clear();
  omg_context_.is_dynamic_input = is_dynamic_input;
  // the default value is ND
  omg_context_.format = domi::DOMI_TENSOR_ND;
  if (!input_format.empty()) {
    std::map<std::string, domi::domiTensorFormat_t>::const_iterator iter =
        ge::input_format_str_to_geformat.find(input_format);
    if (iter != ge::input_format_str_to_geformat.cend()) {
      omg_context_.format = iter->second;
    } else {
      GELOGE(GRAPH_PARAM_INVALID,
             "[Check][Param:InputForamt] %s not support , expect ND/NCHW/NHWC/CHWN/NC1HWC0/NHWC1C0.",
             input_format.c_str());
      return GRAPH_PARAM_INVALID;
    }
  }
  // Input is empty, do not process
  if (input_shape.empty()) {
    return GRAPH_SUCCESS;
  }

  if (!ParseInputShape(input_shape, omg_context_.input_dims, omg_context_.user_input_dims, is_dynamic_input)) {
    GELOGE(GRAPH_PARAM_INVALID, "[Parse][InputShape:InputShape] Failed, shape: %s", input_shape.c_str());
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}
graphStatus Impl::SplitForVariableInferGraph(const ComputeGraphPtr &origin_graph,
                                             const vector<std::string> &const_names,
                                             WeightRefreshableGraphs &weight_refreshable_graphs) const {
  std::vector<ge::NodePtr> const_nodes;
  std::vector<ge::NodePtr> var_nodes;
  GE_ASSERT_SUCCESS(GenerateVariableInferGraph(origin_graph, const_names, weight_refreshable_graphs.infer_graph,
                                                const_nodes, var_nodes));
  GE_ASSERT_SUCCESS(GenerateVariableInitGraph(var_nodes, const_nodes, weight_refreshable_graphs.var_init_graph));
  GE_ASSERT_SUCCESS(GenerateVariableUpdateGraph(var_nodes, weight_refreshable_graphs.var_update_graph));
  return GRAPH_SUCCESS;
}
graphStatus Impl::GenerateVariableInferGraph(const ComputeGraphPtr &origin_graph,
                                             const vector<std::string> &const_names, Graph &infer_graph,
                                             vector<ge::NodePtr> &const_nodes, vector<ge::NodePtr> &var_nodes) const {
  std::string infer_graph_name = "var_infer_graph";
  auto infer_compute_graph = MakeShared<ComputeGraph>(infer_graph_name);
  GE_ASSERT_NOTNULL(infer_compute_graph);
  GE_ASSERT_SUCCESS(GraphUtils::CopyComputeGraph(origin_graph, infer_compute_graph));
  std::unordered_map<std::string, NodePtr> const_nodes_map;
  for (auto &node : infer_compute_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetType() != ge::CONSTANT) {
      continue;
    } else {
      const_nodes_map.insert({node->GetName(), node});
    }
  }
  for (const auto &cur_name : const_names) {
    auto it = const_nodes_map.find(cur_name);
    GE_ASSERT_TRUE((it != const_nodes_map.end()), "can not find const node by name %s", cur_name.c_str());
    GELOGI("find const node %s need to be change to var", cur_name.c_str());
    const_nodes.emplace_back(it->second);
  }
  for (auto &const_node : const_nodes) {
    // get const tensor desc
    const auto &const_op_desc = const_node->GetOpDesc();
    GE_ASSERT_NOTNULL(const_op_desc);
    GeTensorPtr weight;
    (void)AttrUtils::MutableTensor(const_op_desc, ATTR_NAME_WEIGHTS, weight);
    GE_ASSERT_NOTNULL(weight);
    auto const_tensor_desc = weight->GetTensorDesc();

    // construct variable node
    std::string var_name = const_op_desc->GetName() + "_var";
    auto var_node = InsertOp(const_node->GetOwnerComputeGraph(), ge::VARIABLE, var_name,
                             {const_tensor_desc}, {const_tensor_desc});
    GE_ASSERT_NOTNULL(var_node);
    var_nodes.emplace_back(var_node);
    GELOGI("insert variable node %s success", var_node->GetNamePtr());
    // replace const node by variable node
    std::vector<int32_t> output_map(const_node->GetAllOutDataAnchorsSize());
    for (size_t i = 0U; i < const_node->GetAllOutDataAnchorsSize(); ++i) {
      output_map[i] = i;
    }
    GE_ASSERT_SUCCESS(GraphUtils::ReplaceNodeAnchors(var_node, const_node, {}, output_map));
    NodeUtils::UnlinkAll(*const_node);
    GE_ASSERT_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(const_node->GetOwnerComputeGraph(), const_node));
    GELOGI("insert variable node %s replace %s success", var_node->GetNamePtr(), const_node->GetNamePtr());
  }
  infer_graph = GraphUtilsEx::CreateGraphFromComputeGraph(infer_compute_graph);
  return GRAPH_SUCCESS;
}

NodePtr Impl::InsertOp(const ComputeGraphPtr &compute_graph, const string &node_type, const string &node_name,
                       const vector<GeTensorDesc> &input_list, const vector<GeTensorDesc> &output_list) const {
  auto op_desc = MakeShared<OpDesc>(node_name, node_type);
  GE_ASSERT_NOTNULL(op_desc);

  for (const auto &input_desc : input_list) {
    GE_ASSERT_SUCCESS(op_desc->AddInputDesc(input_desc));
  }

  for (const auto &output_desc : output_list) {
    GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(output_desc));
  }

  GE_ASSERT_NOTNULL(compute_graph);
  NodePtr node = compute_graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(node);
  GELOGI("Insert op success, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
  return node;
}

// var_1  const_1   var_2   const_2
//    \     /         \      /
//    assign_1         assign_2
graphStatus Impl::GenerateVariableInitGraph(const vector<ge::NodePtr> &var_nodes,
                                            const vector<ge::NodePtr> &const_nodes,
                                            Graph &init_graph) const {
  GE_ASSERT_TRUE(var_nodes.size() == const_nodes.size());
  std::string init_graph_name = "var_init_graph";
  auto init_compute_graph = MakeShared<ComputeGraph>(init_graph_name);
  GE_ASSERT_NOTNULL(init_compute_graph);
  for (size_t i = 0; i < var_nodes.size(); ++i) {
    auto copy_var_desc = GraphUtils::CopyOpDesc(var_nodes[i]->GetOpDesc());
    GE_ASSERT_NOTNULL(copy_var_desc);
    auto copy_var_node = init_compute_graph->AddNode(copy_var_desc);
    GE_ASSERT_NOTNULL(copy_var_node);
    auto copy_const_desc = GraphUtils::CopyOpDesc(const_nodes[i]->GetOpDesc());
    GE_ASSERT_NOTNULL(copy_const_desc);
    auto copy_const_node = init_compute_graph->AddNode(copy_const_desc);
    GE_ASSERT_NOTNULL(copy_const_node);

    // construct assign node
    std::string assign_name = copy_const_node->GetName() + "_assign";
    GeTensorDesc ref_tensor_desc = copy_var_desc->GetOutputDesc(0);
    GeTensorDesc val_tensor_desc = copy_const_node->GetOpDesc()->GetOutputDesc(0);
    std::vector<GeTensorDesc> input_desc_list = {ref_tensor_desc, val_tensor_desc};
    std::vector<GeTensorDesc> output_desc_list = {ref_tensor_desc};
    auto assign_node = InsertOp(init_compute_graph, ge::ASSIGN, assign_name, input_desc_list, output_desc_list);
    GE_ASSERT_NOTNULL(assign_node);

    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(copy_var_node->GetOutDataAnchor(0), assign_node->GetInDataAnchor(0)));
    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(copy_const_node->GetOutDataAnchor(0), assign_node->GetInDataAnchor(1)));
  }
  init_graph = GraphUtilsEx::CreateGraphFromComputeGraph(init_compute_graph);
  return GRAPH_SUCCESS;
}

//                    +--------then graph--------------+
//   data1     data2  | var_1  data_1   var_2   data_2 |
//     |         |    |   \     /         \      /     |
//    if1       if2   |  assign_1         assign_2     |
//                    +--------------------------------+
graphStatus Impl::GenerateVariableUpdateGraph(const vector<ge::NodePtr> &var_nodes, Graph &update_graph) const {
  std::string update_graph_name = "var_update_graph";
  auto update_compute_graph = MakeShared<ComputeGraph>(update_graph_name);
  GE_ASSERT_NOTNULL(update_compute_graph);
  for (size_t i = 0; i < var_nodes.size(); ++i) {
    GeTensorDesc var_tensor_desc = var_nodes[i]->GetOpDesc()->GetOutputDesc(0);
    // set -2 dim to support empty tensor
    var_tensor_desc.SetUnknownDimNumShape();
    std::string data_name = var_nodes[i]->GetOpDesc()->GetName() + "_data";
    var_tensor_desc.SetOriginShape(GeShape({UNKNOWN_DIM_NUM}));
    auto data_op = InsertOp(update_compute_graph, ge::DATA, data_name, {var_tensor_desc}, {var_tensor_desc});
    GE_ASSERT_NOTNULL(data_op);
    GE_ASSERT_TRUE(AttrUtils::SetInt(data_op->GetOpDesc(), ATTR_NAME_INDEX, i));

    auto if_node = InsertIfNode(var_nodes[i], update_compute_graph);
    GE_ASSERT_NOTNULL(if_node);
    GE_ASSERT_SUCCESS(ConstructIfSubgraphs(var_nodes[i], if_node, update_compute_graph));
    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(data_op->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)));
    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(data_op->GetOutDataAnchor(0), if_node->GetInDataAnchor(1)));
    GELOGI("insert if %s link data %s index %zu successfully", if_node->GetNamePtr(), data_op->GetNamePtr(), i);
  }
  update_graph = GraphUtilsEx::CreateGraphFromComputeGraph(update_compute_graph);
  return GRAPH_SUCCESS;
}

NodePtr Impl::InsertIfNode(const NodePtr &var_node, ComputeGraphPtr &compute_graph) const {
  std::string if_name = var_node->GetOpDesc()->GetName() + "_if";
  OpDescBuilder op_builder(if_name, ge::IF);
  uint32_t input_num = 1U;
  op_builder.AddInput("cond").AddDynamicInput("input", input_num);
  auto if_op_desc = op_builder.Build();
  GE_ASSERT_NOTNULL(if_op_desc);
  if_op_desc->RegisterSubgraphIrName("then_branch", kDynamic);
  if_op_desc->RegisterSubgraphIrName("else_branch", kDynamic);
  auto if_node = compute_graph->AddNode(if_op_desc);
  GE_ASSERT_NOTNULL(if_node);
  return if_node;
}

graphStatus Impl::ConstructIfSubgraphs(const NodePtr &var_node, NodePtr &if_node,
                                       ComputeGraphPtr &compute_graph) const {
  auto if_op_desc = if_node->GetOpDesc();
  std::string then_graph_name = var_node->GetOpDesc()->GetName() + "_then_graph";
  auto then_graph = MakeShared<ComputeGraph>(then_graph_name);
  GE_ASSERT_NOTNULL(then_graph);

  // construct var node
  auto copy_var_desc = GraphUtils::CopyOpDesc(var_node->GetOpDesc());
  auto copy_var_node = then_graph->AddNode(copy_var_desc);
  GE_ASSERT_NOTNULL(copy_var_node);
  // construct inner data node
  std::string then_data_name = copy_var_desc->GetName() + "_then_data";
  GeTensorDesc tensor_desc = copy_var_desc->GetOutputDesc(0);
  auto then_data_node = InsertOp(then_graph, ge::DATA, then_data_name, {tensor_desc}, {tensor_desc});
  GE_ASSERT_NOTNULL(then_data_node);
  GE_ASSERT_TRUE(AttrUtils::SetInt(then_data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1));
  // construct assign node
  std::string assign_name = var_node->GetName() + "_then_assign";
  auto assign_node = InsertOp(then_graph, ge::ASSIGN, assign_name, {tensor_desc, tensor_desc}, {tensor_desc});
  GE_ASSERT_NOTNULL(assign_node);
  GE_ASSERT_SUCCESS(GraphUtils::AddEdge(copy_var_node->GetOutDataAnchor(0), assign_node->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(GraphUtils::AddEdge(then_data_node->GetOutDataAnchor(0), assign_node->GetInDataAnchor(1)));

  GE_ASSERT_SUCCESS(if_op_desc->AddSubgraphName("then_graph"));
  GE_ASSERT_SUCCESS(if_op_desc->SetSubgraphInstanceName(0, then_graph->GetName()));
  then_graph->SetParentGraph(compute_graph);
  then_graph->SetParentNode(if_node);
  GE_ASSERT_SUCCESS(compute_graph->AddSubgraph(then_graph->GetName(), then_graph));
  GELOGI("construct then graph %s successfully", then_graph->GetName().c_str());

  std::string else_graph_name = var_node->GetOpDesc()->GetName() + "_else_graph";
  auto else_graph = MakeShared<ComputeGraph>(else_graph_name);
  GE_ASSERT_NOTNULL(else_graph);
  // construct inner data node
  std::string else_data_name = copy_var_desc->GetName() + "_else_data";
  auto else_data_node = InsertOp(else_graph, ge::DATA, else_data_name, {tensor_desc}, {tensor_desc});
  GE_ASSERT_NOTNULL(else_data_node);
  GE_ASSERT_TRUE(AttrUtils::SetInt(else_data_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1));
  GE_ASSERT_SUCCESS(if_op_desc->AddSubgraphName("else_graph"));
  GE_ASSERT_SUCCESS(if_op_desc->SetSubgraphInstanceName(1, else_graph->GetName()));
  else_graph->SetParentGraph(compute_graph);
  else_graph->SetParentNode(if_node);
  GE_ASSERT_SUCCESS(compute_graph->AddSubgraph(else_graph->GetName(), else_graph));
  GELOGI("construct else graph %s successfully", else_graph->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus aclgrphBuildModel(const ge::Graph &graph, const std::map<std::string, std::string> &build_options,
                              ModelBufferData &model) {
  GELOGD("Enter aclmdlBuildModel process!");
  Impl builder;
  return builder.BuildModel(graph, build_options, model);
}

graphStatus aclgrphBuildModel(const ge::Graph &graph, const std::map<AscendString, AscendString> &build_options,
                              ModelBufferData &model) {
  GELOGD("Enter aclmdlBuildModel process!");
  std::map<std::string, std::string> tmp_build_options;
  for (auto &option : build_options) {
    if (option.first.GetString() == nullptr || option.second.GetString() == nullptr) {
      GELOGE(GRAPH_FAILED, "[Check][Options]AclgrphBuildInitialize option is nullptr.");
      return GRAPH_FAILED;
    }
    std::string key = option.first.GetString();
    std::string val = option.second.GetString();
    tmp_build_options[key] = val;
  }

  Impl builder;
  return builder.BuildModel(graph, tmp_build_options, model);
}

graphStatus CheckVarDesc(const vector<ge::GraphWithOptions> &graph_with_options, const uint64_t session_id) {
  std::map<std::string, GeTensorDesc> var_desc_map;
  for (const auto &graph_options : graph_with_options) {
    auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph_options.graph);
    GE_ASSERT_NOTNULL(compute_graph);
    for (const auto &node : compute_graph->GetAllNodesPtr()) {
      GE_ASSERT_NOTNULL(node);
      if (node->GetType() != ge::VARIABLE) {
        continue;
      }
      auto op_desc = node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      auto out_tensor_desc = op_desc->GetOutputDesc(0);
      const std::string var_name = op_desc->GetName();
      GELOGD("verify var %s, dt %s, format %s", var_name.c_str(),
             TypeUtils::DataTypeToSerialString(out_tensor_desc.GetDataType()).c_str(),
             TypeUtils::FormatToSerialString(out_tensor_desc.GetFormat()).c_str());
      auto it = var_desc_map.find(var_name);
      if (it == var_desc_map.end()) {
        var_desc_map[var_name] = out_tensor_desc;
      } else {
        GE_CHECK_NOTNULL(VarManager::Instance(session_id));
        auto trans_road = VarManager::Instance(session_id)->GetTransRoad(node->GetName());
        bool same_format = (trans_road == nullptr) ? true :
                           (out_tensor_desc.GetFormat() == it->second.GetFormat());
        bool is_same = (same_format && (out_tensor_desc.GetDataType() == it->second.GetDataType()) &&
                        (out_tensor_desc.GetShape() == it->second.GetShape()));
        GE_ASSERT_TRUE(is_same, "var node %s verified fail, current format %s, dt %s, old format %s, dt %s",
                       var_name.c_str(),
                       TypeUtils::FormatToSerialString(out_tensor_desc.GetFormat()).c_str(),
                       TypeUtils::DataTypeToSerialString(out_tensor_desc.GetDataType()).c_str(),
                       TypeUtils::FormatToSerialString(it->second.GetFormat()).c_str(),
                       TypeUtils::DataTypeToSerialString(it->second.GetDataType()).c_str());
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus aclgrphBundleBuildModel(const std::vector<ge::GraphWithOptions> &graph_with_options,
                                    ModelBufferData &model) {
  GELOGD("Enter aclgrphBundleBuildModel process!");
  if (graph_with_options.size() <= 1U) {
    GELOGE(GRAPH_PARAM_INVALID, "graph_with_options size should be larger than 1");
    return GRAPH_PARAM_INVALID;
  }
  std::vector<std::shared_ptr<Impl>> builders;
  SessionId session_id = SessionIdManager::GetNextSessionId();
  auto graph_rebuild_state_ctrl = MakeShared<GraphRebuildStateCtrl>();
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl);
  GE_MAKE_GUARD(destroy_session_resource, [&session_id]() {
    RtContextUtil::GetInstance().DestroyRtContexts(session_id);
    Analyzer::GetInstance()->DestroySessionJsonObject(session_id);
    VarManagerPool::Instance().RemoveVarManager(session_id);
  });
  std::vector<ModelBufferData> models;
  for (size_t i = 0UL; i < graph_with_options.size(); ++i) {
    std::map<std::string, std::string> tmp_build_options;
    for (auto &option : graph_with_options[i].build_options) {
      std::string key = option.first.GetString();
      GE_ASSERT_TRUE(!key.empty());
      std::string val = option.second.GetString();
      GE_ASSERT_TRUE(!val.empty());
      tmp_build_options[key] = val;
    }
    ModelBufferData current_model;
    auto impl = MakeShared<Impl>();
    GE_ASSERT_NOTNULL(impl);
    impl->session_id_ = session_id;
    impl->rebuild_state_ctrl_ = graph_rebuild_state_ctrl;
    GE_ASSERT_SUCCESS(impl->BuildModel(graph_with_options[i].graph, tmp_build_options, current_model));
    builders.emplace_back(std::move(impl));
    models.emplace_back(current_model);
  }
  GE_ASSERT_SUCCESS(CheckVarDesc(graph_with_options, session_id));
  // gather bundle model
  const auto &var_manager = VarManagerPool::Instance().GetVarManager(session_id);
  GE_CHECK_NOTNULL(var_manager);
  uint64_t var_size = static_cast<uint64_t>(var_manager->GetVarMemSize(RT_MEMORY_HBM));
  GE_ASSERT_SUCCESS(ModelHelper::SaveBundleModelBufferToMem(models, var_size, model),
                    "Save models to bundle model failed.");
  return GRAPH_SUCCESS;
}

graphStatus aclgrphConvertToWeightRefreshableGraphs(const ge::Graph &origin_graph,
                                                    const std::vector<AscendString> &const_names,
                                                    WeightRefreshableGraphs &weight_refreshable_graphs) {
  GELOGI("start to execute aclgrphConvertToWeightRefreshableGraphs");
  if (const_names.empty()) {
    GELOGE(GRAPH_PARAM_INVALID, "const_names can not be empty");
    return GRAPH_PARAM_INVALID;
  }
  std::vector<std::string> const_names_tmp;
  std::set<std::string> const_names_tmp_set;
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(origin_graph);
  GE_ASSERT_NOTNULL(compute_graph);
  std::unordered_map<std::string, NodePtr> const_nodes_map;
  for (auto &node : compute_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetType() != ge::CONSTANT) {
      continue;
    } else {
      const_nodes_map.insert({node->GetName(), node});
    }
  }
  for (auto &const_name : const_names) {
    std::string tmp_name = const_name.GetString();
    GE_ASSERT_TRUE(!tmp_name.empty());
    // check const name in graph
    auto it = const_nodes_map.find(tmp_name);
    if (it == const_nodes_map.end()) {
      GELOGE(GRAPH_PARAM_INVALID, "can not find const name %s in graph", tmp_name.c_str());
      return GRAPH_PARAM_INVALID;
    }
    const_names_tmp.emplace_back(tmp_name);
    // check repeated name
    if (!const_names_tmp_set.insert(tmp_name).second) {
      GELOGE(GRAPH_PARAM_INVALID, "can not insert repeat name %s", tmp_name.c_str());
      return GRAPH_PARAM_INVALID;
    }
  }
  Impl builder;
  return builder.SplitForVariableInferGraph(compute_graph, const_names_tmp, weight_refreshable_graphs);
}

static graphStatus aclgrphSaveModelImpl(const std::string &output_file, const ModelBufferData &model) {
  ModelData model_data;
  model_data.model_data = model.data.get();
  model_data.model_len = model.length;
  ModelHelper model_helper;
  const ModelFileHeader *model_header = nullptr;
  auto ret = model_helper.GetModelFileHead(model_data, model_header);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][ModelHeader] Get model header failed.");
    return ret;
  }
  GE_CHECK_NOTNULL(model_header);
  GE_ASSERT_TRUE(model_header->modeltype != static_cast<uint8_t>(MODEL_TYPE_BUNDLE_MODEL),
                 "aclgrphSaveModel does not support saving bundle models, please use aclgrphBundleSaveModel instead.");

  if (model_header->modeltype == ge::MODEL_TYPE_FLOW_MODEL) {
    GELOGE(UNSUPPORTED, "save flow model is not supported.");
    return UNSUPPORTED;
  } else {
    model_helper.SetRepackSoFlag(true);
    GE_ASSERT_SUCCESS(model_helper.LoadRootModel(model_data), "Load root model failed.");
    ModelBufferData model_buffer;
    return model_helper.PackSoToModelData(model_data, output_file + ".om", model_buffer);
  }
  return GRAPH_SUCCESS;
}

graphStatus aclgrphBundleSaveModelImpl(const std::string &output_file, const ModelBufferData &model) {
  ModelData model_data;
  model_data.model_data = model.data.get();
  model_data.model_len = model.length;
  const ModelFileHeader *model_header = nullptr;
  GE_ASSERT_SUCCESS(ModelHelper::GetModelFileHead(model_data, model_header),
                    "[Get][ModelHeader] Get model header failed.");
  GE_CHECK_NOTNULL(model_header);
  GE_ASSERT_TRUE(model_header->modeltype == MODEL_TYPE_BUNDLE_MODEL, "Model is not BundleModel");
  size_t sub_model_num = model_header->model_num;

  std::vector<ModelBufferData> repacked_buffers;
  std::map<std::string, std::pair<int64_t, GeTensorDesc>> var_name_to_verify_info;
  std::string output_file_name = output_file + ".om";
  GE_ASSERT_TRUE(model.length >= (sizeof(ModelFileHeader) + sizeof(ModelPartitionTable)), "Bundle model len is invalid.");
  auto *partition_table = PtrToPtr<uint8_t, ModelPartitionTable>(model.data.get() + sizeof(ModelFileHeader));
  const size_t partition_num = partition_table->num;
  const size_t header_size =
      sizeof(ModelFileHeader) + sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * partition_num;

  size_t current_offset{header_size};
  GE_ASSERT_TRUE(model.length >= current_offset,
      "Bundle model len is invalid, model length is %lu, current_offset is %zu", model.length, current_offset);
  size_t other_part_cnt = 0U; // other part is ordered in front of submodel partition
  for (size_t i = 0UL; i < partition_num; ++i) {
    ModelData sub_model_data;
    sub_model_data.model_data = model.data.get() + current_offset;
    sub_model_data.model_len = partition_table->partition[i].mem_size;
    current_offset += partition_table->partition[i].mem_size;
    GE_ASSERT_TRUE(model.length >= current_offset,
        "Bundle model len is invalid, model length is %lu, current_offset is %zu", model.length, current_offset);
    if (partition_table->partition[i].type != BUNDLE_MODEL_INFO) {
      GELOGI("current partition %zu, type %d is not bundle sub model", i, partition_table->partition[i].type);
      ++other_part_cnt;
      continue;
    }
    ModelHelper model_helper;
    const ModelFileHeader *sub_model_header = nullptr;
    GE_ASSERT_SUCCESS(model_helper.GetModelFileHead(sub_model_data, sub_model_header),
                      "[Get][ModelHeader] Get model header failed.");
    GE_CHECK_NOTNULL(sub_model_header);

    if (sub_model_header->modeltype == ge::MODEL_TYPE_FLOW_MODEL) {
      GELOGE(UNSUPPORTED, "save flow model is not supported.");
      return UNSUPPORTED;
    } else {
      model_helper.SetRepackSoFlag(true);
      GE_ASSERT_SUCCESS(model_helper.LoadRootModel(sub_model_data), "Load root model failed.");
      auto root_model = model_helper.GetGeRootModel();
      GE_ASSERT_NOTNULL(root_model);
      GE_ASSERT_SUCCESS(
          VerifyVarOffset(root_model->GetRootGraph(), var_name_to_verify_info),
          "Variable validation failed. Please ensure that the variables has been compiled under the same session.");
      model_helper.GetGeRootModel()->GetRootGraph();
      GELOGD("Load root model successfully.");
      ModelBufferData cur_buf;
      GE_ASSERT_SUCCESS(
          model_helper.PackSoToModelData(sub_model_data, output_file + ".om", cur_buf, false));
      repacked_buffers.emplace_back(cur_buf);
      if (model_helper.IsSoStore()) {
        output_file_name = ModelHelper::GetOutputFileName();
      }
    }
  }
  GE_ASSERT_TRUE((partition_num - other_part_cnt) == sub_model_num, "partition_num %zu, other_part_cnt %zu, sub_model_num %zu",
                 partition_num, other_part_cnt, sub_model_num);
  GE_ASSERT_TRUE(repacked_buffers.size() == sub_model_num, "repacked num %zu should be equal to %zu",
                 repacked_buffers.size(), sub_model_num);

  // bundle save
  ModelFileHeader *bundle_header = const_cast<ModelFileHeader *>(model_header);
  size_t first_sub_model_offset = partition_table->partition[other_part_cnt].mem_offset; // first sub model offset
  bundle_header->model_length = sizeof(ModelFileHeader) + first_sub_model_offset;
  size_t offset = first_sub_model_offset;
  // only update sub model offset partition info
  for (size_t i = 0UL; i < sub_model_num; ++i) {
    const size_t sub_model_id = i + other_part_cnt;
    partition_table->partition[sub_model_id].mem_size = repacked_buffers[i].length;
    partition_table->partition[sub_model_id].mem_offset = offset;
    bundle_header->model_length += repacked_buffers[i].length;
    offset += repacked_buffers[i].length;
  }
  // save head, partition info and other partition mem
  GE_ASSERT_SUCCESS(FileSaver::SaveToFile(output_file_name, model.data.get(),
                                          (sizeof(ModelFileHeader) + first_sub_model_offset)));
  // save repacked sub models mem last
  for (size_t i = 0UL; i < sub_model_num; ++i) {
    GE_ASSERT_SUCCESS(
        FileSaver::SaveToFile(output_file_name, repacked_buffers[i].data.get(), repacked_buffers[i].length, true));
  }

  GELOGD("Save bundle model [%s] successfully.", output_file_name.c_str());
  return GRAPH_SUCCESS;
}

graphStatus aclgrphSaveModel(const std::string &output_file, const ModelBufferData &model) {
  GELOGD("Enter aclmdlSaveModel process!");
  if (model.data.get() == nullptr || model.length == 0) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][ModelBufferData] model is illegal");
    return GRAPH_PARAM_INVALID;
  }
  return aclgrphSaveModelImpl(output_file, model);
}

graphStatus aclgrphSaveModel(const char_t *output_file, const ModelBufferData &model) {
  GELOGD("Enter aclmdlSaveModel process!");
  if (model.data.get() == nullptr || model.length == 0) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][ModelBufferData]model is illegal");
    return GRAPH_PARAM_INVALID;
  }
  if (output_file == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][OutputFile]file is nullptr.");
    return GRAPH_PARAM_INVALID;
  }
  std::string str_output_file = output_file;
  return aclgrphSaveModelImpl(output_file, model);
}

graphStatus aclgrphBundleSaveModel(const char_t *output_file, const ModelBufferData &model) {
  GELOGD("Enter aclgrphBundleSaveModel process!");
  if (output_file == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][OutputFile]file is nullptr.");
    return GRAPH_PARAM_INVALID;
  }
  std::string output_file_str = output_file;
  return aclgrphBundleSaveModelImpl(output_file_str, model);
}

graphStatus aclgrphGetIRVersion(int32_t *major_version, int32_t *minor_version, int32_t *patch_version) {
  GELOGD("Enter aclgrphGetIRVersion process!");
  GE_CHECK_NOTNULL(major_version);
  GE_CHECK_NOTNULL(minor_version);
  GE_CHECK_NOTNULL(patch_version);
  *major_version = IR_MAJOR_VERSION;
  *minor_version = IR_MINOR_VERSION;
  *patch_version = IR_PATCH_VERSION;
  return GRAPH_SUCCESS;
}

graphStatus aclgrphDumpGraph(const ge::Graph &graph, const char_t *file, const size_t len) {
  GE_CHECK_NOTNULL(file);

  if (len > PATH_MAX || len != strlen(file) || strlen(file) == 0) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][FilePath]file invalid.");
    return GRAPH_PARAM_INVALID;
  }

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  std::string full_path(file, len);
  for (size_t i = 0; i < len; i++) {
    if (full_path[i] == '\\') {
      full_path.replace(i, 1, "/");
    }
  }

  std::string suffix;
  std::string file_path;
  int32_t pos = full_path.rfind("/");
  if (pos != -1) {
    suffix = full_path.substr(pos + 1, -1);
    file_path = full_path.substr(0, pos);
  } else {
    suffix = full_path;
    file_path = "./";
  }

  if (suffix.empty()) {
    suffix = compute_graph->GetName();
    if (suffix.empty()) {
      suffix = "graph";
    }
  }

  char path[PATH_MAX] = {0};
  if (realpath(file_path.c_str(), path) == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][DumpFile] path:%s is invalid.", file);
    return GRAPH_PARAM_INVALID;
  }

  GraphUtils::DumpGEGrph(compute_graph, std::string(path), suffix);
  GraphUtils::DumpGrphToOnnx(*compute_graph, std::string(path), suffix);
  uint64_t i = 0;
  for (const auto &sub_graph_func : compute_graph->GetAllSubgraphs()) {
    auto sub_graph_func_name = suffix + std::string("_sub_graph_") + std::to_string(i++);
    GraphUtils::DumpGEGrph(sub_graph_func, std::string(path), sub_graph_func_name);
    GraphUtils::DumpGrphToOnnx(*sub_graph_func, std::string(path), sub_graph_func_name);
  }

  return GRAPH_SUCCESS;
}

static graphStatus BuildGraphForSigleOp(const OpDescPtr &op_desc, const std::vector<ge::GeTensor> &input_tensors,
                                        const std::vector<ge::GeTensor> &output_tensors, Graph &graph) {
  // call api to get graph
  ge::GeGenerator generator;
  std::string graph_name = ge::CurrentTimeInStr() + "_graph";
  GeGenerator::InOutTensorRef inputs_outputs = {input_tensors, output_tensors};
  std::vector<std::pair<std::string, std::string>> inputs_name_type;
  GE_ASSERT_SUCCESS(generator.BuildSingleOpGraph(op_desc, inputs_outputs, graph_name, graph, inputs_name_type),
                    "[Make][Graph] fail.");
  return GRAPH_SUCCESS;
}

static graphStatus aclgrphGenerateForOp(const OpDescPtr &op_desc, const std::vector<TensorDesc> &inputs,
                                        const std::vector<TensorDesc> &outputs, Graph &graph) {
  // convert input tensordesc to getensor
  std::vector<ge::GeTensor> input_tensors;
  for (const auto &input : inputs) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(input.GetShape().GetDims()), input.GetFormat(), input.GetDataType());

    tensor_desc.SetOriginFormat(input.GetOriginFormat());
    tensor_desc.SetOriginShape(ge::GeShape(input.GetOriginShape().GetDims()));
    ge::TensorUtils::SetRealDimCnt(tensor_desc, static_cast<uint32_t>(input.GetShape().GetDims().size()));
    ge::TensorUtils::SetInputTensor(tensor_desc, true);
    ge::TensorUtils::SetOutputTensor(tensor_desc, false);

    uint8_t *const_data_buffer = nullptr;
    size_t const_data_len = 0;
    (void)input.GetConstData(&const_data_buffer, const_data_len);
    if (const_data_buffer != nullptr) {
      GELOGD("Get const data is not null, will create const input later");
      if (!AttrUtils::SetBool(tensor_desc, ge::CONST_ATTR_NAME_INPUT, true)) {
        GELOGE(ge::FAILED, "[Set][Attr]set attr CONST_ATTR_NAME_INPUT failed.");
        return ge::FAILED;
      }
      ge::ConstGeTensorPtr const_tensor =
          MakeShared<GeTensor>(tensor_desc, const_data_buffer, const_data_len);
      if (const_tensor == nullptr) {
        GELOGE(ge::FAILED, "[Malloc]make shared failed.");
        return ge::FAILED;
      }
      if (!AttrUtils::SetTensor(tensor_desc, ge::ATTR_NAME_WEIGHTS, const_tensor)) {
        GELOGE(ge::FAILED, "[Set][Attr]set attr ATTR_NAME_WEIGHTS failed.");
        return ge::FAILED;
      }
    }

    AscendString input_name;
    GE_CHK_STATUS_RET(input.GetName(input_name), "Get input name failed.");
    if (input_name.GetLength() == 0U) {
      GE_CHK_STATUS_RET(op_desc->AddInputDesc(tensor_desc), "Add input desc failed.");
    } else {
      GE_CHK_STATUS_RET(op_desc->AddInputDesc(input_name.GetString(), tensor_desc), "Add input desc failed.");
    }
    input_tensors.emplace_back(tensor_desc);
  }

  // convert output tensordesc to getensor
  std::vector<ge::GeTensor> output_tensors;
  for (const auto &output : outputs) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(output.GetShape().GetDims()), output.GetFormat(), output.GetDataType());

    tensor_desc.SetOriginFormat(output.GetOriginFormat());
    tensor_desc.SetOriginShape(ge::GeShape(output.GetOriginShape().GetDims()));
    ge::TensorUtils::SetRealDimCnt(tensor_desc, static_cast<uint32_t>(output.GetShape().GetDims().size()));
    ge::TensorUtils::SetInputTensor(tensor_desc, false);
    ge::TensorUtils::SetOutputTensor(tensor_desc, true);

    AscendString output_name;
    GE_CHK_STATUS_RET(output.GetName(output_name), "Get output name failed.");
    if (output_name.GetLength() == 0U) {
      GE_CHK_STATUS_RET(op_desc->AddOutputDesc(tensor_desc), "Add output desc failed.");
    } else {
      GE_CHK_STATUS_RET(op_desc->AddOutputDesc(output_name.GetString(), tensor_desc), "Add output desc failed.");
    }
    output_tensors.emplace_back(tensor_desc);
  }
  return BuildGraphForSigleOp(op_desc, input_tensors, output_tensors, graph);
}

graphStatus aclgrphGenerateForOp(const AscendString &op_type, const std::vector<TensorDesc> &inputs,
                                 const std::vector<TensorDesc> &outputs, Graph &graph) {
  GE_CHECK_NOTNULL(op_type.GetString());
  const auto op_type_str = std::string(op_type.GetString());
  const auto op_name = op_type_str + "_" + std::to_string(ge::GetCurrentTimestamp());
  const auto op_desc = ge::MakeShared<ge::OpDesc>(op_name, op_type_str);
  GE_CHECK_NOTNULL(op_desc);
  return aclgrphGenerateForOp(op_desc, inputs, outputs, graph);
}

graphStatus aclgrphGenerateForOp(const AscendString &json_path, std::vector<Graph> &graphs) {
  std::vector<SingleOpBuildParam> build_params;
  if (SingleOpParser::ParseSingleOpList(json_path.GetString(), build_params) != SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Parse][Singleop] fail.");
    return GRAPH_FAILED;
  }

  for (const auto &param : build_params) {
    Graph graph;
    GE_ASSERT_SUCCESS(BuildGraphForSigleOp(param.op_desc, param.inputs, param.outputs, graph));
    graphs.emplace_back(graph);
  }

  return GRAPH_SUCCESS;
}

static std::string AttrTypeToSerialString(aclgrphAttrType attr_type) {
  auto it = kAttrTypeToStringMap.find(attr_type);
  if (it != kAttrTypeToStringMap.end()) {
    return it->second;
  } else {
    const std::string reason = "aclgrphAttrType " + std::to_string(attr_type) + " is not supported";
    REPORT_PREDEFINED_ERR_MSG("E10055", std::vector<const char_t *>({"reason"}),
                              std::vector<const char_t *>({reason.c_str()}));
    GELOGE(GRAPH_FAILED, "[Check][AclgrphAttrType] attr_type not support %u", attr_type);
    return "UNDEFINED";
  }
}

graphStatus aclgrphSetOpAttr(Graph &graph, aclgrphAttrType attr_type, const char_t *cfg_path) {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  if (cfg_path == nullptr) {
    return GRAPH_SUCCESS;
  }

  auto iter = kAttrTypeFuncMap.find(attr_type);
  if (iter == kAttrTypeFuncMap.end()) {
    GELOGE(GRAPH_FAILED,
           "[Check][AclgrphAttrType]%s is not supported. Valid attr_type is: "
           "[0:ATTR_TYPE_KEEP_DTYPE][1:ATTR_TYPE_WEIGHT_COMPRESS]",
           AttrTypeToSerialString(attr_type).c_str());
    return GRAPH_FAILED;
  }

  std::string path = cfg_path;
  return iter->second(compute_graph, path);
}

}  // namespace ge

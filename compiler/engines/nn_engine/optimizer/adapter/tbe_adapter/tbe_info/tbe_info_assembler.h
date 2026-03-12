/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_TBE_INFO_ASSEMBLER_H_
#define FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_TBE_INFO_ASSEMBLER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "adapter/common/get_attr_by_type.h"
#include "common/aicore_util_types.h"
#include "common/fe_op_info_common.h"
#include "ops_store/op_kernel_info.h"
#include "tensor_engine/tbe_op_info.h"
#include "platform/platform_info.h"
#include "common/sgt_slice_type.h"

namespace fe {
using ToOpStructPtr = std::shared_ptr<ToOpStruct_t>;
struct AllPlatFormInfo {
  PlatFormInfos platform_info;

  OptionalInfos opti_compilation_info;
};

struct TensorDescAndIndex {
  ge::GeTensorDescPtr tensor_desc_ptr;
  string name_in_op_kernel;
  size_t index_in_op_kernel;
  uint32_t index_in_opdesc;
  bool is_input;
  ge::Format propagat_heavy_format = ge::FORMAT_RESERVED;
  int32_t propagat_sub_format = 0;
  bool is_first_layer_conv = false;
  TensorDescAndIndex(const ge::GeTensorDescPtr &tensor_desc_ptr_param, const string &name_in_op_kernel_param,
                     size_t index_in_op_kernel_param, uint32_t index_in_opdesc_param, bool is_input_param,
                     ge::Format propagat_heavy_format_param = ge::FORMAT_RESERVED,
                     int32_t propagat_sub_format_param = 0, bool is_first_layer_conv_param = false) :
        tensor_desc_ptr(tensor_desc_ptr_param),
        name_in_op_kernel(name_in_op_kernel_param),
        index_in_op_kernel(index_in_op_kernel_param),
        index_in_opdesc(index_in_opdesc_param),
        is_input(is_input_param),
        propagat_heavy_format(propagat_heavy_format_param),
        propagat_sub_format(propagat_sub_format_param),
        is_first_layer_conv(is_first_layer_conv_param) {}
};

using SetConstValueWithDtype = std::function<Status(ge::GeTensorPtr, const std::string &, te::TbeOpTensor &)>;

using SetConstValueWithDtypePtr = std::shared_ptr<SetConstValueWithDtype>;

Status CreateTbeTensor(const ge::OpDesc &op_desc, const TensorDescAndIndex &tensor_info, te::TbeOpTensor &tbe_tensor);

class TbeInfoAssembler {
 public:
  Status Initialize();

  Status AssembleTbeInfo(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                         const std::string &engine_name, te::TbeOpInfo &tbe_op_info);

  Status AssembleTbeInfo(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                         const HeavyFormatInfo &heavy_format_info, const std::string &engine_name,
                         te::TbeOpInfo &tbe_op_info);

  Status AssembleTbeInfo(ge::Node *node, const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &tbe_op_info,
                         const string &engine_name);

  Status AssembleAutoFuseTbeInfo(ge::Node *node, te::TbeOpInfo &tbe_op_info) const;

  /*
   *  @ingroup fe
   *  @brief   set Attrs to tbe_op_info
   *  @param   [in]  op              op desc
   *  @param   [in]  op_kernel_info_ptr op kernel info
   *  @param   [in/out]  op_info      tbe data item
   *  @return  SUCCESS or FAILED
   */
  Status FeedAttrsToTbeOpInfo(const ge::OpDesc &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
                              te::TbeOpInfo &op_info) const;
  /*
   *  @ingroup fe
   *  @brief   set Attrs:flagint64 to tbe_op_info
   *  @param   [in]  node            input node pointer
   *  @param   [in/out]  op_info      tbe data item
   *  @return  SUCCESS or FAILED
   */
  Status FeedFlagInt64ToTbeOpInfo(const ge::Node *node, te::TbeOpInfo &op_info) const;

  /*
   *  @ingroup fe
   *  @brief   set is_unknown_shape to tbe_op_info
   *  @param   [in]  op              op desc
   *  @param   [in/out]  op_info      tbe data item
   *  @return  SUCCESS or FAILED
   */
  Status FeedIsUnknownShapeToTbeOpInfo(const ge::OpDesc &op_desc, te::TbeOpInfo &op_info) const;

  /*
   *  @ingroup fe
   *  @brief   convert output tensors of node to tbe_op_info object
   *  @param   [in]  node            input node pointer
   *  @param   [in]  output_map       output name
   *  @param   [in]  op_kernel_info_ptr tensor from const node
   *  @param   [in/out]  op_info      tbe data item
   *  @return  SUCCESS or FAILED
   */
  Status FeedOutputsToTbeOpInfo(const ge::Node *node, IndexNameMap &output_idx_name_map,
                                OpKernelInfoPtr op_kernel_info_ptr, te::TbeOpInfo &op_info) const;

  /*
   *  @ingroup fe
   *  @brief   set inputs to tbe_op_info
   *  @param   [in]  node            input node pointer
   *  @param   [in]  input_map        input name
   *  @param   [in]  op_kernel_info_ptr tensor from const node
   *  @param   [in/out]  op_info      tbe data item
   *  @return  SUCCESS or FAILED
   */
  Status FeedInputsToTbeOpInfo(const ge::Node *node, IndexNameMap &input_idx_name_map,
                               OpKernelInfoPtr op_kernel_info_ptr, te::TbeOpInfo &op_info) const;

  Status FeedAutoFuseInoutputsToTbeOpInfo(const ge::Node *node, const bool is_input,
                                         te::TbeOpInfo &tbe_op_info) const;

  /*
   *  @ingroup fe
   *  @brief   set parameter infos to tbe_op_info
   *  @param   [in]      node           op node pointer
   *  @param   [in]      input_info_ptr   op info store pointer
   *  @param   [in/out]  input          global temp param
   *  @param   [in]      input_tensor    tensor from const node
   *  @param   [in/out]  op_info         tbe data item
   *  @param   [in]      input_size     number of inputs or outputs of op
   *  @param   [in]      i             index to input
   *  @param   [in]      is_input_or_output input or output
   *  @return  SUCCESS or FAILED
   */
  Status FeedParameterInfoForInput(const ge::Node *node, const InputOrOutputInfoPtr &info_ptr, int index_in_opdesc,
                                   bool last_item_flag, te::TbeOpTensor &tbe_op_tensor, te::TbeOpParam &tbe_op_param,
                                   te::TbeOpInfo &tbe_op_info) const;

  Status FeedParameterInfoForOutput(const ge::OpDesc &op_desc, const ge::GeTensorDesc &output_desc,
                                    const InputOrOutputInfoPtr &info_ptr, bool last_item_flag,
                                    te::TbeOpTensor &tbe_op_tensor, te::TbeOpParam &tbe_op_param,
                                    te::TbeOpInfo &tbe_op_info) const;

  Status FeedParameterInfoForNotFound(const InputOrOutputInfoPtr &info_ptr, const string &is_input_or_output,
                                      te::TbeOpParam &tbe_op_param, te::TbeOpInfo &tbe_op_info) const;

  /*
   *  @ingroup fe
   *  @brief   convert input tensor desc object of op to tbe op info object
   *  @param   [in]  op_desc               op desc
   *  @param   [in]  input_idx_name_map    index and name mapping of input
   *  @param   [in/out] op_kernel_info_ptr kernel info of op
   *  @param   [in/out] op_info            tbe op info object
   *  @return  SUCCESS or FAILED
   */

  Status ConvertInputsToTbeOpInfo(const ge::OpDesc &op_desc, IndexNameMap &input_idx_name_map,
                                  OpKernelInfoPtr op_kernel_info_ptr, const HeavyFormatInfo &heavy_format_info,
                                  te::TbeOpInfo &op_info) const;

  Status ConvertInputsToTbeOpInfo(const ge::NodePtr &node, IndexNameMap &input_idx_name_map,
                                  OpKernelInfoPtr op_kernel_info_ptr, const HeavyFormatInfo &heavy_format_info,
                                  te::TbeOpInfo &op_info) const;

  /*
   *  @ingroup fe
   *  @brief   convert output tensor desc object of op to tbe op info object
   *  @param   [in]  op_desc               op desc
   *  @param   [in]  output_idx_name_map   index and name mapping of output
   *  @param   [in/out] op_kernel_info_ptr kernel info of op
   *  @param   [in/out] op_info            tbe op info object
   *  @return  SUCCESS or FAILED
   */
  Status ConvertOutputsToTbeOpInfo(const ge::NodePtr &node, IndexNameMap &output_idx_name_map,
                                   OpKernelInfoPtr op_kernel_info_ptr, const HeavyFormatInfo &heavy_format_info,
                                   te::TbeOpInfo &op_info) const;

  Status ConvertParameterInfoForInput(InputOrOutputInfoPtr info_ptr, te::TbeOpParam &input,
                                      te::TbeOpTensor &input_tensor, te::TbeOpInfo &op_info, bool last_item_flag) const;

  void FeedL1InputTensor(const ToOpStructPtr &l1_info, const ge::OpDescPtr &op_desc,
                         IndexNameMap &input_idx_name_map, const uint32_t &index_in_opdesc,
                         te::TbeOpTensor &input_tensor) const;
  void FeedL2InputTensor(const ToOpStructPtr &l2_info, const ge::OpDescPtr &op_desc,
                         IndexNameMap &input_idx_name_map, const uint32_t &index_in_opdesc,
                         te::TbeOpTensor &input_tensor) const;

  Status SetInputTensorBaseInfo(const ge::OpDescPtr &op_desc, const uint32_t &index_in_opdesc,
                                te::TbeOpTensor &input_tensor) const;

  void FeedFusionOutputTensor(const ToOpStructPtr &fusion_info, const ge::OpDescPtr &op_desc, IndexNameMap &output_idx_name_map,
                              const uint32_t &index_in_opdesc, te::TbeOpTensor &output_tensor) const;

  void GetOpInputL1Attr(const ge::OpDescPtr &op_desc, std::vector<int64_t> &op_input_l1_flag,
                        std::vector<int64_t> &op_input_l1_addr, std::vector<int64_t> &op_input_l1_valid_size) const;

  Status JudgeShapeToSetFlag(const ge::OpDescPtr &op_desc, const bool& is_input,
                             te::TbeOpInfo &op_info, bool &flag) const;

  void SetInplaceAttr(const ge::OpDescPtr &op_desc, const IndexNameMap &input_map,
                      const IndexNameMap &output_map) const;

  map<std::string, std::string> GetAllOptionsForTBE(const ge::OpDesc &op_desc,
                                                    const string &engine_name,
                                                    const OpKernelInfoPtr &op_kernel_info_ptr);

  void SetExtraParams(const ge::OpDesc &op_desc, te::TbeOpInfo &op_info) const;

  void SetHashedExtraParams(const ge::OpDesc &op_desc, te::TbeOpInfo &op_info) const;
  
  void SetCustCoreNum(const ge::OpDesc &op_desc, te::TbeOpInfo &op_info) const;
  
  void SetSingleOpScene(const ge::Node *node, te::TbeOpInfo &op_info) const;

  void SetOpDynamicRank(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const;

  void SetOpStorePattern(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const;

  void SetOpImplSwitch(const ge::OpDescPtr &op_desc_ptr, te::TbeOpInfo &tbe_op_info) const;

  void SetOpJitCompile(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const;

  void SetOpImplMode(const std::string &engine_name, const ge::OpDescPtr &op_desc_ptr, te::TbeOpInfo &op_info) const;

  void SetNeedPreCompile(const ge::OpDescPtr &op_desc_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                         te::TbeOpInfo &op_info) const;

  void SetOpDebugConfig(const std::string &engine_name, const ge::OpDescPtr &op_desc_ptr, te::TbeOpInfo &op_info) const;

 private:
  bool SetOutputRealIndexInplaceAttr(const vector<vector<int64_t>> &output_inplace,
                                     std::map<size_t, std::pair<size_t, size_t>> &input_ir_real_index,
                                     std::map<size_t, std::pair<size_t, size_t>> &output_ir_real_index,
                                     vector<vector<int64_t>> &real_output_inplace) const;

  void TransIOIrIndxToRealIndex(const std::vector<size_t> &output_real_idex,
                                const std::vector<size_t> &input_real_idex,
                                vector<vector<int64_t>> &real_output_inplace) const;

  void GetPrivateAttrsList(const ge::OpDesc &op_desc, std::vector<te::TbeAttrValue> &private_attrs_list) const;
                           
  void GenerateTbePrivateAttrValue(const ge::OpDesc &op_desc,
                                   const ge::AnyValue &value_type,
                                   te::TbeAttrValue &tbe_attr_value, const string &attr_name) const;

  void CalibrateCoreNum(const ge::OpDesc &op_desc, const string &engine_name,
                        const OpKernelInfoPtr &op_kernel_info_ptr,
                        map<std::string, std::string> &options);

  void SetL1Info(te::TbeOpTensor &input_tensor, const std::vector<int64_t> &op_input_l1_flag,
                 const std::vector<int64_t> &op_input_l1_addr, const std::vector<int64_t> &op_input_l1_valid_size,
                 const uint32_t &index_in_opdesc) const;

  Status GetSpecificIndex(const ge::OpDesc &op_desc, const IndexNameMap &name_map,
                          const std::string &input_name_in_op_kernel, bool is_input,
                          vector<uint32_t> &specific_input_index) const;

  Status FindAndCheckEndNodeForConstValue(const ge::Node *node, const uint32_t &tensor_index,
                                          InputOrOutputInfoPtr tensor_info_ptr,
                                          ge::NodePtr &other_end_node, bool &is_const_node) const;

  void SetInputDdrBaseProp(const ge::Node *node, const uint32_t &tensor_index, te::TbeOpTensor &input_tensor) const;

  void SetOutputDdrBaseProp(const ge::Node *node, const uint32_t &tensor_index, te::TbeOpTensor &output_tensor) const;

  void SetIsConstInputFlag(const ge::Node *node, const ge::OpDesc &op_desc,
                           const uint32_t &tensor_index, te::TbeOpTensor &input_tensor) const;

  void SetIsNullOutputFlag(const ge::Node *node, const uint32_t &tensor_index, te::TbeOpTensor &output_tensor) const;

  Status SetTensorConstValue(const ge::Node *node, const uint32_t &tensor_index, InputOrOutputInfoPtr tensor_info_ptr,
                             te::TbeOpTensor &op_tensor) const;
  Status AssembleConstValue(ge::GeTensorPtr const_tensor_ptr, const ge::OpDescPtr &op_desc,
                            te::TbeOpTensor &op_tensor) const;

  void SetTbeInfoLimitedRange(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const;

  void SetTbeInfoVectorCore(const ge::OpDescPtr op_desc_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                            te::TbeOpInfo &op_info) const;

  void SetOutputInplaceAttr(ge::OpDescPtr &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr) const;

  void TransStringToVector2D(const string &outputInplaceAbility, vector<vector<int64_t>> &res);

  static Status SetConstValueWithFloat16(ge::GeTensorPtr tensor_ptr, const std::string &tensor_name,
                                         te::TbeOpTensor &op_tensor);

  static Status SetConstValueWithBf16(ge::GeTensorPtr tensor_ptr, const std::string &tensor_name,
                                      te::TbeOpTensor &op_tensor);

  template <typename T>
  static Status SetConstValue(ge::GeTensorPtr tensor_ptr, const std::string &tensor_name, te::TbeOpTensor &op_tensor);

  template <typename T>
  static void GetConstValueVec(ge::GeTensorPtr &const_tensor_ptr, vector<T> &const_data_vec);

  void FindAmplifiedCoreNum(uint64_t exec_time,
                            string &final_core_num_str);

  bool NeedCalibration(const string &engine_name,
                       map<std::string, std::string> &options,
                       string &default_core_num_str);

  void FeedOutputTensorAtomicAttr(const ffts::ThreadSliceMapPtr &slice_info, te::TbeOpTensor &output_tensor,
                                  const uint32_t &index_in_opdesc) const;
  AllPlatFormInfo all_plat_info_;

  static const std::map<ge::DataType, SetConstValueWithDtypePtr> set_const_value_func_map;

  /* <Initial block dim, <time(ns), amplified block dim>> */
  static const std::map<uint32_t, std::vector<std::pair<uint64_t, string>>> time_to_core_num_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_TBE_INFO_ASSEMBLER_H_

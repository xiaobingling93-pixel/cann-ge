/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_FE_INNER_ATTR_DEFINE_H_
#define FUSION_ENGINE_UTILS_COMMON_FE_INNER_ATTR_DEFINE_H_

#include <string>
#include "common/aicore_util_attr_define.h"

namespace fe {
const std::string kGlobalWorkspaceRef = "globalworkspace_ref";

// sgt
const std::string TBE_OP_THREAD_ATOMIC_OUTPUT_INDEX = "tbe_op_thread_atomic_output_index";

const std::string TBE_OP_THREAD_ATOMIC_WORKSPACE_FLAG = "tbe_op_thread_atomic_workspace_flag";

const std::string TBE_OP_THREAD_ATOMIC_DTYPES = "tbe_op_thread_atomic_dtypes";

const std::string TBE_OP_THREAD_ATOMIC_INT64_VALUES = "tbe_op_thread_atomic_int64_values";

const std::string TBE_OP_THREAD_ATOMIC_UINT64_VALUES = "tbe_op_thread_atomic_uint64_values";

const std::string TBE_OP_THREAD_ATOMIC_FLOAT_VALUES = "tbe_op_thread_atomic_float_values";

const std::string TBE_OP_THREAD_ATOMIC_WSP_MODE = "tbe_op_thread_atomic_wsp_mode";

const std::string ATTR_NAME_THREAD_COMPRESS_PARAMETERS = "thread_compress_parameters";

const std::string ATTR_NAME_THREAD_WEIGHT_REPEAT = "_thread_weight_repeat";

const std::string ATTR_NAME_SGT_SLICE_SHAPE = "_sgt_slice_shape";

const std::string ATTR_NAME_SGT_ORI_SLICE_SHAPE = "_sgt_ori_slice_shape";
// end sgt
const std::string PERM = "perm";

const std::string CONCAT_DIM = "concat_dim";

const std::string SPLIT_DIM = "split_dim";

const std::string AIPP_CONV_FLAG = "Aipp_Conv_Flag";

const std::string ATTR_NAME_CACHE_READ_MODE = "cache_read_mode";

const std::string IS_CHECK_SUPPORTED = "isCheckSupported";

const std::string INFER_FORMAT = "infer_format";

const std::string INPUT_ND_TO_OTHER_FORMAT = "_input_nd_to_other_format";

const std::string KEEP_DIMS = "keep_dims";

const std::string CUSTOM_OP_IMPL_CONFIG_PATH = "_custom_op_impl_config_path";

const std::string ATTR_NAME_COMPRESS_PARAMETERS = "compress_parameters";

const std::string ATTR_NAME_WEIGHT_COMPRESS_TYPE = "weight_compress_type";

const std::string ATTR_NAME_FE_GROUP = "_fe_group";

const std::string ATTR_NAME_FE_PROPAGAT_HEAVY_FORMAT = "_fe_propagat_heavvy_format";

const std::string ATTR_NAME_GROUPS = "groups";

const std::string ATTR_NAME_WEIGHT_REPEAT = "_weight_repeat";

const std::string ATTR_NAME_RESHAPE_CXVALUE = "_fe_reshape_caxisvalue";

const std::string ATTR_NAME_FE_WEIGHT_COMPRESS = "_fe_weight_compress";

const std::string ATTR_NAME_WEIGHT_COMPRESS = "_weight_compress";
const std::string ATTR_NAME_COMPRESS_TYPE_FLAG = "alg";

const std::string ATTR_NAME_DTYPE_IS_UPDATED = "dtype_is_updated";

const std::string FORMAT_AGNOSTIC = "_format_agnostic";

const std::string TBE_ATOMIC_KERNEL = "tbeAtomicKernel";

const std::string ATTR_NAME_IS_CONNECTED_TO_NETOUTPUT = "is_connected_to_netoutput";

const std::string IS_GE_OP = "_is_ge_op";

const std::string INPUT_FORMAT_AGNOSTIC_EXCEPTION = "_format_agnostic_except_input";

const std::string OUTPUT_FORMAT_AGNOSTIC_EXCEPTION = "_format_agnostic_except_output";

const std::string IS_CUSTOM_OP = "_is_custom_op";

const std::string FUSION_FAILED_ID_ATTR = "fusion_failed";

const std::string NON_PERSISTENT_CUSTOM_OP_FLAG = "_custom_op_flag";

/* This attr IS_FIRST_LAYER_CONV_FOR_OP is for inference Conv2D only.
 * FE will pass this attr to conv2d.py.
 * When this attr is true, Conv2D will only return C04. */
const std::string IS_FIRST_LAYER_CONV_FOR_OP = "_is_first_layer_conv_for_op";

const std::string IS_FIRST_LAYER_CONV = "_is_first_layer_conv";

const std::string ATTR_STRIDE_ATTR_STRIDE = "stride";

const std::string FORMAT_CONTINUOUS = "_format_continuous";

const std::string REFRESH_CONTINUOUS_FLAG = "_refresh_continuous_flag";

const std::string KEEP_DTYPE = "_keep_dtype";

const std::string OP_IMPL_MODE_ENUM = "_op_impl_mode_enum";

const std::string OP_CUSTOM_IMPL_MODE_ENUM = "_op_custom_impl_mode_enum";

const std::string ROLLBACK_IF_FAILED = "_rollback_if_failed";

const std::string ATTR_NAME_PARENT_NODE = "parentNode";

const std::string ATTR_NAME_IS_COMPIED_FUSION_OP = "_is_compiled_fusion_op";

const std::string ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";

const std::string kAttrAtomicOpRunInfo = "atomic_op_run_info";

const std::string ATTR_NAME_INPUT_IS_VAR = "INPUT_IS_VAR";
const std::string ATTR_NAME_OUTPUT_IS_VAR = "OUTPUT_IS_VAR";

const std::string kOpPrecisionModeStr = "op_precision_mode_str";

const std::string kIsComeFromConstOp = "_is_come_from_const_op";

const std::string kIsLimitedGraph = "_is_limited_graph";

const std::string kAttrKernelBinId = "_kernel_bin_id";

const std::string kAttrMemsetKernelBinId = "_memset_kernel_bin_id";

const std::string kThreadId = "_thread_id";

const std::string kSliceInstanceNum = "_slice_instance_num";

const std::string kOriginalNode = "_original_node";

const std::string kAttrHeadFilePath = "_head_file_path";

const std::string kAttrInheritDtypeFromPredecessor = "_inherit_dtype_from_predecessor";

const std::string kAttrSupported_16In_32Out = "_supported_16in_32out";

const std::string kAttrThread1Node = "_ffts_thread_1_node";

const std::string kAttrRelatedThreadsNodes = "_ffts_other_thread_nodes";

const std::string kTilingRemoveDuplicates = "op_unique_key";

const std::string kAttrNameIsHeavyOp = "_is_heavy_op";

const std::string kAttrScheduleMode = "_soft_sync_schedule_mode";

const std::string kAttrWeightPrefetchNodeName = "_weight_prefetch_node_name";

const std::string kAttrWeightPrefetchType = "_weight_prefetch_type";

const std::string kAttrWeightPrefetchSrcOffset = "_weight_prefetch_src_offset";

const std::string kAttrWeightPrefetchDstOffset = "_weight_prefetch_dst_offset";

const std::string kAttrWeightPrefetchDataSize = "_weight_prefetch_data_size";

const std::string kSingleOpUbPassNameAttr = "_single_op_ub_pass_name";

const std::string kAclnnLoweringFunc = "AclnnLoweringFunc";

const std::string kPrecisionModeEnum = "_precision_mode_enum";

const std::string kMustPromoteFlag = "_must_promote_type";

const std::string kPromoteInfo = "_promote_info";

const std::string kAttrDumpAble = "_dump_able";

const std::string kAttrTilingDataStr = "_tiling_data_str";

const std::string kAttrTileFwkOpStr = "_tile_kwk_op";

const std::string kAttrPrefixStr = "_prefix";

const std::string kAttrSubkernelOpBinaryStr = "_subkernel_op_binary";

const std::string kAicpuBlockDim = "_aicpu_blockdim";

const std::string ATTR_NAME_IS_NULL_OUTPUT = "_is_null_output";
}
#endif  // FUSION_ENGINE_UTILS_COMMON_FE_INNER_ATTR_DEFINE_H_

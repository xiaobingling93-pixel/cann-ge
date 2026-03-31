/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_api_registry.h"

namespace codegen {
namespace {
class Register {
 public:
  Register();
};

Register::Register() {
  const std::string kAscendcCastRegStr = {
#include "cast_reg_base.h"
  };
  const std::string kAscendcCompareRegStr = {
#include "compare_reg_base.h"
  };
  const std::string kAscendcConcatRegBaseStr = {
#include "concat_reg_base.h"
  };
  const std::string kAscendcDatacopyRegBaseStr = {
#include "datacopy_reg_base.h"
  };
  const std::string kAscendcDatacopyNddmaRegBaseStr = {
#include "datacopy_nddma_reg_base.h"
  };
  const std::string kAscendcReduce_initRegBase = {
#include "reduce_init_reg_base.h"
  };
  const std::string kAscendcFloorDivRegBaseStr = {
#include "floor_div_reg_base.h"
  };
  const std::string kAscendcSignRegBaseStr = {
#include "sign_reg_base.h"
  };
  const std::string kAscendcWhereRegBaseStr = {
#include "where_reg_base.h"
  };
  const std::string kAscendcWhereV2RegBaseStr = {
#include "where_v2_reg_base.h"
  };
  const std::string kAscendcGatherRegBaseStr = {
#include "gather_reg_base.h"
  };
  const std::string kAscendcUtilsRegBaseStr = {
#include "utils_reg_base.h"
  };
  const std::string kAscendcBroadcastRegStr = {
#include "broadcast_reg_base.h"
  };
  const std::string kAscendcLogicalNotStr = {
#include "logical_not_reg_base.h"
  };
  const std::string kAscendcClipByValueRegStr = {
#include "clipbyvalue_reg_base.h"
  };
  const std::string kAscendcLogicalRegBaseStr = {
#include "logical_reg_base.h"
  };
  const std::string kAscendcPowRegBaseStr = {
#include "pow_reg_base.h"
  };
  const std::string kAscendcExp2RegBaseStr = {
#include "exp2_reg_base.h"
  };
  const std::string kAscendcLog1pRegBaseStr = {
#include "log1p_reg_base.h"
  };
  const std::string kAscendcErfRegBaseStr = {
#include "erf_reg_base.h"
  };
  const std::string kAscendcTanhRegBaseStr = {
#include "tanh_reg_base.h"
  };
  const std::string kAscendcSubRegBaseStr = {
#include "sub_reg_base.h"
  };
  const std::string kAscendcDivRegBaseStr = {
#include "div_reg_base.h"
  };
  const std::string kAscendcSplitRegBaseStr = {
#include "split_reg_base.h"
  };
  const std::string kAscendcAtan2RegBaseStr = {
#include "atan2_reg_base.h"
  };
  const std::string kAscendcCopySignRegBaseStr = {
#include "copy_sign_reg_base.h"
  };
  const std::string kAscendcErfcxRegBaseStr = {
#include "erfcx_reg_base.h"
  };
  const std::string kAscendcExpmRegBaseStr = {
#include "expm_reg_base.h"
  };
  const std::string kAscendcTruncDivRegBaseStr = {
#include "trunc_div_reg_base.h"
  };
  const std::string kAscendcRemainderRegBaseStr = {
#include "remainder_reg_base.h"
  };
  std::unordered_map<std::string, std::string> api_to_file{
      {"cast_reg_base.h", kAscendcCastRegStr},
      {"compare_reg_base.h", kAscendcCompareRegStr},
      {"concat_reg_base.h", kAscendcConcatRegBaseStr},
      {"datacopy_reg_base.h", kAscendcDatacopyRegBaseStr},
      {"datacopy_nddma_reg_base.h", kAscendcDatacopyNddmaRegBaseStr},
      {"pow_reg_base.h", kAscendcPowRegBaseStr},
      {"exp2_reg_base.h", kAscendcExp2RegBaseStr},
      {"log1p_reg_base.h", kAscendcLog1pRegBaseStr},
      {"erf_reg_base.h", kAscendcErfRegBaseStr},
      {"tanh_reg_base.h", kAscendcTanhRegBaseStr},
      {"reduce_init_reg_base.h", kAscendcReduce_initRegBase},
      {"floor_div_reg_base.h", kAscendcFloorDivRegBaseStr},
      {"sign_reg_base.h", kAscendcSignRegBaseStr},
      {"where_reg_base.h", kAscendcWhereRegBaseStr},
      {"where_v2_reg_base.h", kAscendcWhereV2RegBaseStr},
      {"gather_reg_base.h", kAscendcGatherRegBaseStr},
      {"broadcast_reg_base.h", kAscendcBroadcastRegStr},
      {"utils_reg_base.h", kAscendcUtilsRegBaseStr},
      {"logical_not_reg_base.h", kAscendcLogicalNotStr},
      {"clipbyvalue_reg_base.h", kAscendcClipByValueRegStr},
      {"logical_reg_base.h", kAscendcLogicalRegBaseStr},
      {"split_reg_base.h", kAscendcSplitRegBaseStr},
      {"sub_reg_base.h", kAscendcSubRegBaseStr},
      {"div_reg_base.h", kAscendcDivRegBaseStr},
      {"atan2_reg_base.h", kAscendcAtan2RegBaseStr},
      {"copy_sign_reg_base.h", kAscendcCopySignRegBaseStr},
      {"erfcx_reg_base.h", kAscendcErfcxRegBaseStr},
      {"expm_reg_base.h", kAscendcExpmRegBaseStr},
      {"trunc_div_reg_base.h", kAscendcTruncDivRegBaseStr},
      {"remainder_reg_base.h", kAscendcRemainderRegBaseStr},
  };

  AscendCApiRegistry::GetInstance().RegisterApi(api_to_file);
}

Register __attribute__((unused)) reg_base_api_register;
}  // namespace
}  // namespace codegen
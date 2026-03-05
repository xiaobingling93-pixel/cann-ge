/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_ATC_UTILS_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_ATC_UTILS_H_
#include <gtest/gtest.h>
#include "api/atc/cmd_flag_info.h"
#include "common/context/local_context.h"

// 这些选项的值是定义在main_impl.cc的全局变量，全局变量的值会影响到每个用例的执行，所以如果用例里面有
// 新用到的选项，需要在TearDown()函数里面恢复到main_impl.cc中定义的默认值； 注；TearDown是每个用例执行
// 完都会执行一次
DECLARE_int32(mode);
DECLARE_string(output);
DECLARE_int32(framework);
DECLARE_string(out_nodes);
DECLARE_string(input_format);
DECLARE_string(output_type);
DECLARE_string(input_shape);
DECLARE_string(dynamic_batch_size);
DECLARE_string(dynamic_image_size);
DECLARE_string(dynamic_dims);
DECLARE_string(singleop);
DECLARE_string(auto_tune_mode);
DECLARE_string(log);
DECLARE_string(om);
DECLARE_string(json);
DECLARE_string(soc_version);
DECLARE_string(model);
DECLARE_string(dump_mode);
DECLARE_string(insert_op_conf);
DECLARE_string(display_model_info);
DECLARE_string(keep_dtype);
DECLARE_string(is_input_adjust_hw_layout);
DECLARE_string(is_output_adjust_hw_layout);
DECLARE_string(input_fp16_nodes);
DECLARE_string(precision_mode);
DECLARE_string(precision_mode_v2);
DECLARE_string(op_name_map);
DECLARE_string(op_precision_mode);
DECLARE_string(modify_mixlist);
DECLARE_string(compress_weight_conf);
DECLARE_string(enable_compress_weight);
DECLARE_string(weight);
DECLARE_string(log);
DECLARE_string(op_select_implmode);
DECLARE_string(optypelist_for_implmode);
DECLARE_string(allow_hf32);
DECLARE_string(compression_optimize_conf);
DECLARE_string(input_hint_shape);
DECLARE_string(check_report);
DECLARE_int32(virtual_type);
DECLARE_string(external_weight);
DECLARE_string(enable_small_channel);
DECLARE_string(quant_dumpable);
DECLARE_string(ac_parallel_enable);
DECLARE_string(tiling_schedule_optimize);
DECLARE_string(atomic_clean_policy);
DECLARE_string(status_check);
DECLARE_string(input_shape_range);

// Atc基类，涉及atc入口的可以继承此类
class AtcTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {
    ge::OmgContext ctx;
    domi::GetContext() = ctx; // ctx全局对象需要重置
    ge::flgs::GetUserOptions().clear();
    FLAGS_mode = 0;
    FLAGS_output.clear();
    FLAGS_framework = -1;
    FLAGS_auto_tune_mode.clear();
    FLAGS_out_nodes.clear();
    FLAGS_input_format.clear();
    FLAGS_output_type.clear();
    FLAGS_input_shape.clear();
    FLAGS_dynamic_batch_size.clear();
    FLAGS_singleop.clear();
    FLAGS_auto_tune_mode.clear();
    FLAGS_log = "null";
    FLAGS_om.clear();
    FLAGS_json.clear();
    FLAGS_soc_version.clear();
    FLAGS_model.clear();
    FLAGS_dump_mode = "0";
    FLAGS_insert_op_conf = "";
    FLAGS_display_model_info = "0";
    FLAGS_keep_dtype = "";
    FLAGS_is_input_adjust_hw_layout = "false";
    FLAGS_is_output_adjust_hw_layout = "false";
    FLAGS_input_fp16_nodes = "";
    FLAGS_precision_mode = "";
    FLAGS_op_name_map = "";
    FLAGS_dynamic_image_size = "";
    FLAGS_dynamic_dims = "";
    FLAGS_op_precision_mode = "";
    FLAGS_modify_mixlist = "";
    FLAGS_compress_weight_conf = "";
    FLAGS_enable_compress_weight = "";
    FLAGS_weight = "";
    FLAGS_log = "null";
    FLAGS_op_select_implmode = "";
    FLAGS_optypelist_for_implmode = "";
    FLAGS_allow_hf32 = "";
    FLAGS_compression_optimize_conf = "";
    FLAGS_input_hint_shape.clear();
    FLAGS_check_report = "check_result.json";
    FLAGS_virtual_type = 0;
    FLAGS_external_weight = "0";
    FLAGS_enable_small_channel = "0";
    FLAGS_quant_dumpable = "";
    FLAGS_ac_parallel_enable = "0";
    FLAGS_tiling_schedule_optimize = "0";
    FLAGS_atomic_clean_policy = "0";
    FLAGS_status_check = "0";
    FLAGS_input_shape_range = "";
  }
};
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNNING_ENV_INCLUDE_GE_RUNNING_ENV_ATC_UTILS_H_

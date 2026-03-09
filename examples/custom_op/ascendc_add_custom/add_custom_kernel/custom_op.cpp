/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <fstream>
#include <cstddef>
#include <cstdint>
#include "register/op_def_registry.h"
#include "graph/tensor.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "graph/custom_op.h"
#include "add_custom_kernel.h"
#include "tiling/tiling_api.h"
#include "tiling/matrix/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

using namespace ge;

/**
 * AscendC自定义加法算子实现类
 * 继承EagerExecuteOp，实现NPU端自定义Add算子的前向执行逻辑，支持广播语义的形状推导和多数据类型计算
 */
class AddCustom : public EagerExecuteOp {
public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) {
    // 获取算子输入Tensor
    const gert::Tensor *input_x = ctx->GetInputTensor(0);
    const gert::Tensor *input_y = ctx->GetInputTensor(1);
    if (input_x == nullptr || input_y == nullptr) {
      std::cerr << "Input tensor is null!" << std::endl;
      return GRAPH_FAILED;
    }
    // 申请输出内存
    const gert::StorageShape &output_shape = input_x->GetShape();
    uint32_t tensor_size = input_x->GetSize();
    DataType data_type = input_x->GetDataType();
    const gert::StorageFormat &format = input_x->GetFormat();
    // 申请输出Tensor内存并校验
    gert::Tensor *output_z = ctx->MallocOutputTensor(0, output_shape, format, data_type, tensor_size);
    if (output_z == nullptr) {
      std::cerr <<"Failed to malloc output tensor memory!"<< std::endl;
      return GRAPH_FAILED;
    }
    // 获取需处理的元素个数和 grid
    int64_t n_elements = input_x->GetShapeSize();
    // 核函数实现中指定的一次性处理的数据块大小
    const int32_t BLOCK_SIZE_VALUE = 1024;

    int32_t grid_x = std::ceil(static_cast<double>(n_elements) / (BLOCK_SIZE_VALUE));
    int32_t grid_y = 1;
    int32_t grid_z = 1;
    uint32_t num_block = grid_x * grid_y * grid_z;

    void *z_addr = output_z->GetAddr();
    void *x_addr = const_cast<void*>(input_x->GetAddr());
    void *y_addr = const_cast<void*>(input_y->GetAddr());
    // 启动核函数执行加法计算
    aclrtStream stream = ctx->GetStream();
    // 调用封装函数
    launch_add_custom(
            static_cast<uint8_t*>(x_addr),
            static_cast<uint8_t*>(y_addr),
            static_cast<uint8_t*>(z_addr),
            tensor_size,
            num_block,
            stream
    );
    return GRAPH_SUCCESS;
    }
};

/**
 * 注册AscendC自定义加法算子
 * 配置输入输出类型、形状推导函数、数据类型推导规则
 */
REG_OP(AddCustom)
.INPUT(x, "T")
.INPUT(y, "T")
.OUTPUT(z, "T")
.DATATYPE(T, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
.OP_END_FACTORY_REG(AddCustom);

/**
 * 加法算子形状推导函数
 * 实现PyTorch风格的广播语义
 */
graphStatus InferShapeForAdd(gert::InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    auto min_num = std::min(input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    auto max_num = std::max(input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    if (min_num != 1) {
      std::cerr <<"Add param invalid" << std::endl;
    } else {
      if (input_shape_1.GetDimNum() > 1) {
        *output_shape = input_shape_1;
      } else {
        *output_shape = input_shape_0;
      }
      return GRAPH_SUCCESS;
    }
  }
  //处理标量输入
  if (input_shape_0.GetDimNum() == 0) {
    *output_shape = input_shape_1;
    return GRAPH_SUCCESS;
  }
  if (input_shape_1.GetDimNum() == 0) {
    *output_shape = input_shape_0;
    return GRAPH_SUCCESS;
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
  }
  return GRAPH_SUCCESS;
}

IMPL_OP(AddCustom).InferShape(InferShapeForAdd);

REG_AUTO_MAPPING_OP(AddCustom);
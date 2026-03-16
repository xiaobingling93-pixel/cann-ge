/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <fstream>
#include "graph/custom_op.h"
#include "acl/acl_rt.h"

using namespace ge;
/*
 *
 * @brief 从二进制文件（npubin）中读取数据
 * @param [in] filePath: file path
 * @return 装有二进制文件数据的 buffer
 */
std::vector<char> LoadNpubin(const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file) {
    return {};
  }
  file.seekg(0, std::ios::end);
  const size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (size == 0) {
    return {};
  }

  std::vector<char> buffer(size);
  file.read(buffer.data(), size);

  if (!file && !file.eof()) {
    return {};
  }
  return buffer;
}
/*
AddCustom:用户的自定义 Triton 算子的实现类，重点实现 Execute 函数
主要流程：
1. 从 npubin 中获取 kernel 句柄
2. 构造 kernel args
  2.1 获取 input Tensor 相关
  2.2 申请 output 内存
  2.3 拼接 args 参数
3. 调用 acl 接口 launch kernel
*/
class AddCustom : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) {
    const char *bin_path = "./add_kernel.npubin";
    // 从 npubin 中读取二进制数据
    std::vector<char> bin_data = LoadNpubin(bin_path);
    if (bin_data.empty()) {
      std::cerr << __FILE__ << ":" << __LINE__ << "Load Bin Errors" << std::endl;
      return GRAPH_FAILED;
    }
    // 从二进制数据中获取 Kernel 句柄
    aclrtBinHandle bin_handle = nullptr;
    aclrtFuncHandle func_handle = nullptr;
    aclrtBinaryLoadOption binary_load_option;
    aclrtBinaryLoadOptions binary_load_options;

    binary_load_option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    binary_load_option.value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;

    binary_load_options.numOpt = 1;
    binary_load_options.options = &binary_load_option;

    aclError ret = ACL_ERROR_NONE;
    ret = aclrtBinaryLoadFromData(bin_data.data(), bin_data.size(), &binary_load_options, &bin_handle);
    if (ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError: " << ret << std::endl;
      return GRAPH_FAILED;
    }

    ret = aclrtBinaryGetFunction(bin_handle, "add_kernel", &func_handle);
    if (ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError: " << ret << std::endl;
      return GRAPH_FAILED;
    }

    // 获取算子输入 Tensor
    const gert::Tensor *input_x = ctx->GetInputTensor(0);
    const gert::Tensor *input_y = ctx->GetInputTensor(1);

    // 申请输出内存
    const gert::StorageShape &output_shape = input_x->GetShape();
    size_t tensor_size = input_x->GetSize();
    DataType data_type = input_x->GetDataType();
    const gert::StorageFormat &format = input_x->GetFormat();
    gert::Tensor *output_z = ctx->MallocOutputTensor(0, output_shape, format, data_type);
    void *z_addr = output_z->GetAddr();

    // 获取需处理的元素个数和 grid
    int64_t n_elements = input_x->GetShapeSize();
    // 核函数实现中指定的一次性处理的数据块大小
    const int32_t BLOCK_SIZE_VALUE = 1024;
    // 向量加法按照1维网格的方式分块处理，因此需要计算grid_x的值，grid_y/grid_z默认为1
    int32_t grid_x = std::ceil(static_cast<double>(n_elements) / (BLOCK_SIZE_VALUE));
    int32_t grid_y = 1;
    int32_t grid_z = 1;
    int32_t block_num = grid_x * grid_y * grid_z;

    // 拼装 args
    // args 的前3个参数和后3个参数是固定的，中间的是用户自定义的，要求是和核函数的签名里的变量顺序以及类型严格一致
    // 按照当前的样例中的 Kernel 实现，要求 AddCustom 的两个输入的shape必须相同，不支持BroadCast的方式，例如 Shape1 = [2,3,4],Shape2=[4],这种就不支持
    struct __attribute__((packed)) {
      // void *ffts_addr __attribute__((aligned(8)));  // 注意：如果设备是A3，则需要加上这个参数
      void *sync_block_lock __attribute__((aligned(8)));
      void *workspace_addr __attribute__((aligned(8)));
      const void *arg0 __attribute__((aligned(8)));
      const void *arg1 __attribute__((aligned(8)));
      void *arg2 __attribute__((aligned(8)));
      int32_t arg3 __attribute__((aligned(4)));
      int32_t grid_x __attribute__((aligned(4)));
      int32_t grid_y __attribute__((aligned(4)));
      int32_t grid_z __attribute__((aligned(4)));
    } args = {
        // nullptr, // 如果设备是 A3 则需要传递这个参数
        nullptr, nullptr, input_x->GetAddr(), input_y->GetAddr(), z_addr, static_cast<int32_t>(n_elements),
        grid_x,  grid_y,  grid_z,
    };
    // 获取 stream
    void *stream = ctx->GetStream();
    // launch Kernel
    // 函数原型：
    // aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle func_handle,
    //                                      uint32_t block_num,
    //                                      aclrtStream stream,
    //                                      aclrtLaunchKernelCfg *cfg,
    //                                      void *hostArgs,
    //                                      size_t size,
    //                                      aclrtPlaceHolderInfo *placeHolderArray,
    //                                      size_t placeHolderNum);
    // func_handle[in]:核函数句柄
    // block_num  [in]:指定核函数在几个核上运行
    // stream     [in]:任务 stream
    // cfg        [in]:任务下发的配置信息，不需要可填 nullptr
    // hostArgs   [in]:核函数参数地址
    // size       [in]:参数所占字节数大小
    // placeHolderArray[in]:占位数组
    // placeHolderNum[in]:占位数组的个数
    ret = aclrtLaunchKernelWithHostArgs(func_handle, static_cast<uint32_t>(block_num), stream, nullptr,
                                        static_cast<void *>(&args), sizeof(args), nullptr, 0);
    if (ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError: " << ret << std::endl;
      return GRAPH_FAILED;
    }
    ret = aclrtBinaryUnLoad(bin_handle);
    if (ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError: " << ret << std::endl;
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
};
REG_AUTO_MAPPING_OP(AddCustom);

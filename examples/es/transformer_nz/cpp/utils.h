/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _UTILS_H_
#define _UTILS_H_
#include <numeric>
#include <sstream>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <map>
#include "graph/tensor.h"
#include "ge/es_graph_builder.h"
namespace ge {
class Utils {
 public:

  template<typename Iterator>
  static std::string Join(Iterator begin, Iterator end, const std::string &sep) {
    if (begin == end) {
      return "";
    }
    std::stringstream ss;
    ss << *begin;
    for (auto iter = std::next(begin); iter != end; ++iter) {
      ss << sep << *iter;
    }
    return ss.str();
  }

  template<typename T>
  static void PrintTensorToConsole(const ge::Tensor &tensor) {
    auto shape = tensor.GetTensorDesc().GetShape();
    auto dims = shape.GetDims();
    // 处理标量情况
    if (dims.empty()) {
      std::cout << "tensor shape: scalar" << std::endl;
    } else {
      std::cout << "tensor shape: " << Join(dims.begin(), dims.end(), ",") << std::endl;
    }

    std::cout << "tensor data:  ";
    auto data_cnt = shape.GetShapeSize();
    auto data_begin = reinterpret_cast<const T *>(tensor.GetData());

    // 处理标量或单元素情况
    if (data_cnt <= 1) {
      std::cout << *data_begin << std::endl;
    } else {
      std::cout << *data_begin;
      for (auto data = std::next(data_begin); data != data_begin + data_cnt; ++data) {
        std::cout << ", " << *data;
      }
      std::cout << std::endl;
    }
  }

  template<typename T>
  static void PrintTensorToFile(const ge::Tensor &tensor, const std::string &prefix, int64_t index) {
    std::string filename = prefix + "_" + std::to_string(index) + ".data";
    std::ofstream data_file(filename);
    auto shape = tensor.GetTensorDesc().GetShape();
    auto dims = shape.GetDims();
    // 处理标量情况
    if (dims.empty()) {
      data_file << "tensor shape: scalar" << std::endl;
    } else {
      data_file << "tensor shape: " << Join(dims.begin(), dims.end(), ",") << std::endl;
    }

    data_file << "tensor data:  ";
    auto data_cnt = shape.GetShapeSize();
    auto data_begin = reinterpret_cast<const T *>(tensor.GetData());

    // 处理标量或单元素情况
    if (data_cnt <= 1) {
      data_file << *data_begin << std::endl;
    } else {
      data_file << *data_begin;
      for (auto data = std::next(data_begin); data != data_begin + data_cnt; ++data) {
        data_file << ", " << *data;
      }
      data_file << std::endl;
    }
    data_file.close();

    // 打印保存信息
    std::cout << prefix << "[" << index << "] save to file " << filename << std::endl;
  }

  template<typename T>
  static std::unique_ptr<ge::Tensor> StubTensor(const std::vector<T> &data,
                                                const std::vector<int64_t> &shape,
                                                Format format = FORMAT_ND) {
    if constexpr (std::is_same_v<T, float>) {
      return ge::es::CreateTensor<T>(data.data(), shape.data(), shape.size(), DT_FLOAT, format);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return ge::es::CreateTensor<T>(data.data(), shape.data(), shape.size(), DT_INT32, format);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return ge::es::CreateTensor<T>(data.data(), shape.data(), shape.size(), DT_INT64, format);
    } else {
      std::cout << "unsupported type: " << typeid(T).name() << std::endl;
      return nullptr;
    }
  }
  static void PrintTensorsToConsole(const std::vector<ge::Tensor> &tensors) {
    for (const auto &tensor : tensors) {
      auto data_type = tensor.GetTensorDesc().GetDataType();
      switch (data_type) {
        case ge::DT_FLOAT:PrintTensorToConsole<float>(tensor);
          break;
        case ge::DT_INT32:PrintTensorToConsole<int32_t>(tensor);
          break;
        case ge::DT_INT64:PrintTensorToConsole<int64_t>(tensor);
          break;
        default:std::cout << "unsupported type: " << static_cast<int64_t>(data_type) << std::endl;
          break;
      }
    }
  }
  static void PrintTensorsToFile(const std::vector<ge::Tensor> &tensors, const std::string &prefix = "tensor") {
    static std::map<std::string, int64_t> index_map;
    int64_t &index = index_map[prefix];
    for (const auto &tensor : tensors) {
      auto data_type = tensor.GetTensorDesc().GetDataType();
      switch (data_type) {
        case ge::DT_FLOAT:PrintTensorToFile<float>(tensor, prefix, index);
          break;
        case ge::DT_INT32:PrintTensorToFile<int32_t>(tensor, prefix, index);
          break;
        case ge::DT_INT64:PrintTensorToFile<int64_t>(tensor, prefix, index);
          break;
        default:std::cout << "unsupported type: " << static_cast<int64_t>(data_type) << std::endl;
          break;
      }
      index++;
    }
  }
};
}
#endif //_UTILS_H_
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <vector>
#include "acl/acl.h"
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
    if (data_begin == nullptr) {
      std::cout << "<null>" << std::endl;
      return;
    }

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
    if (data_begin == nullptr) {
      data_file << "<null>" << std::endl;
      data_file.close();
      std::cout << prefix << "[" << index << "] save to file " << filename << std::endl;
      return;
    }

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

  static size_t GetElementCount(const std::vector<int64_t> &shape) {
    if (shape.empty()) {
      return 1U;
    }
    size_t count = 1U;
    for (const auto dim : shape) {
      if (dim <= 0) {
        return 0U;
      }
      count *= static_cast<size_t>(dim);
    }
    return count;
  }

  static size_t GetDataTypeSize(const ge::DataType dt) {
    switch (dt) {
      case ge::DT_FLOAT:
        return sizeof(float);
      case ge::DT_INT32:
        return sizeof(int32_t);
      case ge::DT_INT64:
        return sizeof(int64_t);
      default:
        return 0U;
    }
  }

  template<typename T>
  static bool CreateDeviceInputTensor(const std::vector<T> &host_data,
                                      const std::vector<int64_t> &shape,
                                      const ge::DataType dt,
                                      ge::Tensor &device_tensor) {
    const size_t element_count = GetElementCount(shape);
    if (element_count == 0U || host_data.size() != element_count) {
      return false;
    }
    const size_t bytes = element_count * sizeof(T);
    ge::TensorDesc desc(ge::Shape(shape), FORMAT_ND, dt);
    desc.SetPlacement(ge::kPlacementDevice);
    device_tensor = ge::Tensor(desc);

    uint8_t *device_ptr = nullptr;
    const aclError malloc_ret =
        aclrtMalloc(reinterpret_cast<void **>(&device_ptr), bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (malloc_ret != ACL_SUCCESS || device_ptr == nullptr) {
      return false;
    }
    const aclError memcpy_ret =
        aclrtMemcpy(device_ptr, bytes, host_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (memcpy_ret != ACL_SUCCESS) {
      (void)aclrtFree(reinterpret_cast<void *>(device_ptr));
      return false;
    }
    auto free_device_ptr = [](uint8_t *ptr) {
      if (ptr != nullptr) {
        (void)aclrtFree(reinterpret_cast<void *>(ptr));
      }
    };
    if (device_tensor.SetData(device_ptr, bytes, free_device_ptr) != ge::GRAPH_SUCCESS) {
      (void)aclrtFree(reinterpret_cast<void *>(device_ptr));
      return false;
    }
    return device_tensor.SetPlacement(ge::kPlacementDevice) == ge::GRAPH_SUCCESS;
  }

  static bool CreateDeviceOutputTensor(const std::vector<int64_t> &shape,
                                       const ge::DataType dt,
                                       ge::Tensor &device_tensor) {
    const size_t element_count = GetElementCount(shape);
    const size_t dt_size = GetDataTypeSize(dt);
    if (element_count == 0U || dt_size == 0U) {
      return false;
    }
    const size_t bytes = element_count * dt_size;
    ge::TensorDesc desc(ge::Shape(shape), FORMAT_ND, dt);
    desc.SetPlacement(ge::kPlacementDevice);
    device_tensor = ge::Tensor(desc);

    uint8_t *device_ptr = nullptr;
    const aclError malloc_ret =
        aclrtMalloc(reinterpret_cast<void **>(&device_ptr), bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (malloc_ret != ACL_SUCCESS || device_ptr == nullptr) {
      return false;
    }
    auto free_device_ptr = [](uint8_t *ptr) {
      if (ptr != nullptr) {
        (void)aclrtFree(reinterpret_cast<void *>(ptr));
      }
    };
    if (device_tensor.SetData(device_ptr, bytes, free_device_ptr) != ge::GRAPH_SUCCESS) {
      (void)aclrtFree(reinterpret_cast<void *>(device_ptr));
      return false;
    }
    return device_tensor.SetPlacement(ge::kPlacementDevice) == ge::GRAPH_SUCCESS;
  }

  static bool CopyDeviceOutputsToHost(const std::vector<ge::Tensor> &device_outputs,
                                      std::vector<ge::Tensor> &host_outputs) {
    host_outputs.clear();
    host_outputs.reserve(device_outputs.size());
    for (const auto &device_tensor : device_outputs) {
      auto host_desc = device_tensor.GetTensorDesc();
      host_desc.SetPlacement(ge::kPlacementHost);
      ge::Tensor host_tensor(host_desc);
      const size_t bytes = device_tensor.GetSize();
      if (bytes != 0U) {
        std::vector<uint8_t> host_data(bytes);
        const aclError memcpy_ret = aclrtMemcpy(host_data.data(), bytes, device_tensor.GetData(), bytes,
                                                ACL_MEMCPY_DEVICE_TO_HOST);
        if (memcpy_ret != ACL_SUCCESS) {
          return false;
        }
        if (host_tensor.SetData(host_data) != ge::GRAPH_SUCCESS) {
          return false;
        }
      }
      host_outputs.emplace_back(host_tensor);
    }
    return true;
  }

  static bool CopyTensorToHostIfNeed(const ge::Tensor &tensor, ge::Tensor &host_tensor) {
    const auto placement = tensor.GetTensorDesc().GetPlacement();
    if (placement != ge::kPlacementDevice) {
      host_tensor = tensor;
      return true;
    }
    auto host_desc = tensor.GetTensorDesc();
    host_desc.SetPlacement(ge::kPlacementHost);
    host_tensor = ge::Tensor(host_desc);
    const size_t bytes = tensor.GetSize();
    if (bytes == 0U) {
      return true;
    }
    std::vector<uint8_t> host_data(bytes);
    const aclError memcpy_ret = aclrtMemcpy(host_data.data(), bytes, tensor.GetData(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (memcpy_ret != ACL_SUCCESS) {
      return false;
    }
    return host_tensor.SetData(host_data) == ge::GRAPH_SUCCESS;
  }

  static void PrintTensorsToConsole(const std::vector<ge::Tensor> &tensors) {
    for (const auto &tensor : tensors) {
      ge::Tensor host_tensor;
      if (!CopyTensorToHostIfNeed(tensor, host_tensor)) {
        std::cout << "copy tensor to host failed when printing to console" << std::endl;
        continue;
      }
      auto data_type = host_tensor.GetTensorDesc().GetDataType();
      switch (data_type) {
        case ge::DT_FLOAT:PrintTensorToConsole<float>(host_tensor);
          break;
        case ge::DT_INT32:PrintTensorToConsole<int32_t>(host_tensor);
          break;
        case ge::DT_INT64:PrintTensorToConsole<int64_t>(host_tensor);
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
      ge::Tensor host_tensor;
      if (!CopyTensorToHostIfNeed(tensor, host_tensor)) {
        std::cout << prefix << "[" << index << "] copy tensor to host failed when printing to file" << std::endl;
        index++;
        continue;
      }
      auto data_type = host_tensor.GetTensorDesc().GetDataType();
      switch (data_type) {
        case ge::DT_FLOAT:PrintTensorToFile<float>(host_tensor, prefix, index);
          break;
        case ge::DT_INT32:PrintTensorToFile<int32_t>(host_tensor, prefix, index);
          break;
        case ge::DT_INT64:PrintTensorToFile<int64_t>(host_tensor, prefix, index);
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

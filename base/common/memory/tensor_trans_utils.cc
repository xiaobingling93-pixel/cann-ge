/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_trans_utils.h"
#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/util/mem_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "runtime/rt.h"
#include "formats/utils/formats_trans_utils.h"
#include "graph/utils/type_utils.h"
#include "rt_error_codes.h"
namespace ge {
namespace {
constexpr size_t kValAlignment = 64U;
constexpr uint64_t kMemAlignSize = 32U;

std::string RtTensorDescToString(const gert::Tensor &rt_tensor) {
  std::stringstream ss;
  ss << " [";
  ss << "shape:[" << formats::GertShapeToString(rt_tensor.GetShape().GetStorageShape()) << "],";
  ss << "origin_shape:[" << formats::GertShapeToString(rt_tensor.GetShape().GetOriginShape()) << "],";
  ss << "format:[" << TypeUtils::FormatToSerialString(rt_tensor.GetFormat().GetStorageFormat()) << "],";
  ss << "origin_format:[" << TypeUtils::FormatToSerialString(rt_tensor.GetFormat().GetOriginFormat()) << "],";
  ss << "dtype:[" << TypeUtils::DataTypeToSerialString(rt_tensor.GetDataType()) << "]";
  ss << "]";
  return ss.str();
}

std::string GeTensorDescToString(const GeTensorDesc &tensor_desc) {
  std::stringstream ss;
  ss << " [";
  ss << "shape:[" << formats::ShapeToString(tensor_desc.GetShape()) << "],";
  ss << "origin_shape:[" << formats::ShapeToString(tensor_desc.GetOriginShape()) << "],";
  ss << "format:[" << TypeUtils::FormatToSerialString(tensor_desc.GetFormat()) << "],";
  ss << "origin_format:[" << TypeUtils::FormatToSerialString(tensor_desc.GetOriginFormat()) << "],";
  ss << "dtype:[" << TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()) << "]";
  ss << "]";
  return ss.str();
}
template <class T>
class TensorWrapper {
  static_assert(std::is_same_v<T, ge::GeTensor>, "T must be ge::GeTensor");
public:
  explicit TensorWrapper(std::shared_ptr<T> tensor) : tensor_(std::move(tensor)) {}

  [[nodiscard]] const std::shared_ptr<T> &GetTensor() const { return tensor_; }

  // executor not support multi thread, so same addr can not exist in different threads
  static graphStatus Manager(gert::TensorAddress const addr, gert::TensorOperateType operate_type, void **out) {
    GE_ASSERT_NOTNULL(addr);
    if (operate_type == gert::TensorOperateType::kFreeTensor) {
      auto tensor_wrapper = reinterpret_cast<TensorWrapper<T> *>(addr);
      tensor_wrapper->Free();
      if (tensor_wrapper->GetCount() == 0U) {
        delete tensor_wrapper;
      }
      return GRAPH_SUCCESS;
    }
    if (operate_type == gert::TensorOperateType::kGetTensorAddress) {
      auto tensor_wrapper = reinterpret_cast<TensorWrapper<T> *>(addr);
      GE_ASSERT_NOTNULL(out);
      GE_ASSERT_NOTNULL(tensor_wrapper);
      GE_ASSERT_NOTNULL(tensor_wrapper->GetTensor());
      *out = PtrToPtr<uint8_t, void>(tensor_wrapper->GetTensor()->MutableData().GetData());
      return GRAPH_SUCCESS;
    }
    if (operate_type == gert::TensorOperateType::kPlusShareCount) {
      auto tensor_wrapper = reinterpret_cast<TensorWrapper<T> *>(addr);
      (void)tensor_wrapper->AddCount();
      return GRAPH_SUCCESS;
    }
    GELOGE(ge::PARAM_INVALID, "Unexpected operate type %d", static_cast<int32_t>(operate_type));
    return GRAPH_FAILED;
  }
  void Free() {
    if (GetCount() > 0U) {
      if (SubCount() == 0U) {
        tensor_ = nullptr;
      }
    }
  }
  size_t AddCount() {
    if (count_ < std::numeric_limits<size_t>::max()) {
      ++count_;
    }
    return count_;
  }
  size_t SubCount() {
    if (count_ > 0) {
      --count_;
    }
    return count_;
  }
  size_t GetCount() const {
    return count_;
  }
private:
  std::shared_ptr<T> tensor_;
  size_t count_ = 1U;
};

std::vector<int64_t> GetDimsFromGertShape(const gert::Shape &gert_shape) {
  std::vector<int64_t> dims;
  for (size_t i = 0U; i < gert_shape.GetDimNum(); ++i) {
    (void)dims.emplace_back(gert_shape.GetDim(i));
  }
  return dims;
}

gert::Shape GetGertShapeFromDims(const SmallVector<int64_t, kDefaultDimsNum> &dims) {
  gert::Shape shape;
  shape.SetDimNum(dims.size());
  for (size_t i = 0U; i < dims.size(); i++) {
    shape.SetDim(i, dims[i]);
  }
  return shape;
}

gert::Shape GetGertShapeFromTensor(const Tensor &ge_tensor) {
  gert::Shape shape;
  shape.SetDimNum(ge_tensor.GetShapeDimNum());
  for (size_t i = 0U; i < shape.GetDimNum(); i++) {
    shape.SetDim(i, ge_tensor.GetShapeDim(i));
  }
  return shape;
}

TensorDesc GetTensorDescFromGertTensor(const gert::Tensor &gert_tensor) {
  Shape storage_shape{GetDimsFromGertShape(gert_tensor.GetStorageShape())};
  TensorDesc tensor_desc{storage_shape, gert_tensor.GetStorageFormat(), gert_tensor.GetDataType()};
  const Shape origin_shape{GetDimsFromGertShape(gert_tensor.GetOriginShape())};

  tensor_desc.SetOriginFormat(gert_tensor.GetOriginFormat());
  tensor_desc.SetOriginShape(origin_shape);
  const auto placement = gert::TensorPlacementUtils::IsOnHost(gert_tensor.GetPlacement()) ?
    ge::Placement::kPlacementHost : ge::Placement::kPlacementDevice;
  tensor_desc.SetPlacement(placement);
  return tensor_desc;
}

ge::GeTensorDesc GetInnerTensorDescFromGertTensor(const gert::Tensor &gert_tensor) {
  ge::GeShape storage_shape{GetDimsFromGertShape(gert_tensor.GetStorageShape())};
  ge::GeTensorDesc tensor_desc{storage_shape, gert_tensor.GetStorageFormat(), gert_tensor.GetDataType()};
  const ge::GeShape origin_shape{GetDimsFromGertShape(gert_tensor.GetOriginShape())};

  tensor_desc.SetOriginFormat(gert_tensor.GetOriginFormat());
  tensor_desc.SetOriginShape(origin_shape);
  tensor_desc.SetOriginDataType(gert_tensor.GetDataType());
  const auto placement = gert::TensorPlacementUtils::IsOnHost(gert_tensor.GetPlacement()) ?
    ge::Placement::kPlacementHost : ge::Placement::kPlacementDevice;
  tensor_desc.SetPlacement(placement);
  return tensor_desc;
}
} // namespace

GeShape TensorTransUtils::ContructGeShapeFromRtShape(const gert::Shape &rt_shape) {
  std::vector<int64_t> dims(rt_shape.GetDimNum());
  for (size_t i = 0U; i < rt_shape.GetDimNum(); ++i) {
    dims[i] = rt_shape.GetDim(i);
  }
  return GeShape(dims);
}

gert::Shape TensorTransUtils::ContructRtShapeFromShape(const Shape &ge_shape) {
  gert::Shape rt_shape;
  rt_shape.SetDimNum(ge_shape.GetDimNum());
  for (size_t i = 0U; i < ge_shape.GetDimNum(); ++i) {
    rt_shape.SetDim(i, ge_shape.GetDim(i));
  }
  return rt_shape;
}

gert::Shape TensorTransUtils::ContructRtShapeFromGeShape(const GeShape &ge_shape) {
  gert::Shape rt_shape;
  rt_shape.SetDimNum(ge_shape.GetDimNum());
  for (size_t i = 0U; i < ge_shape.GetDimNum(); ++i) {
    rt_shape.SetDim(i, ge_shape.GetDim(i));
  }
  return rt_shape;
}

gert::Shape TensorTransUtils::ContructRtShapeFromVector(const std::vector<int64_t> &dims) {
  gert::Shape rt_shape;
  rt_shape.SetDimNum(dims.size());
  for (size_t i = 0U; i < dims.size(); ++i) {
    rt_shape.SetDim(i, dims[i]);
  }
  return rt_shape;
}

std::vector<int64_t> TensorTransUtils::GetDimsFromGertShape(const gert::Shape &gert_shape) {
  std::vector<int64_t> dims;
  dims.reserve(gert_shape.GetDimNum());
  for (size_t i = 0U; i < gert_shape.GetDimNum(); ++i) {
    (void)dims.emplace_back(gert_shape.GetDim(i));
  }
  return dims;
}

Status TensorTransUtils::TransHostGertTensorsToDevice(Allocator *allocator,
    const std::vector<gert::Tensor> &src_tensors, std::vector<gert::Tensor> &dst_tensors,
    std::vector<MemBlock *> &inputs_memblocks, bool enable_input_batch_cpy) {
  MemcpyBatchParam memcpy_batch_param;
  int32_t device_id = -1;
  (void)rtGetDevice(&device_id);
  memcpy_batch_param.device_id = device_id;
  size_t attr_idx = 0;
  GE_ASSERT_TRUE(dst_tensors.empty(), "dst_tensors is not empty");
  dst_tensors.resize(src_tensors.size());
  for (size_t i = 0U; i < src_tensors.size(); ++i) {
    size_t aligned_size = 0U;
    dst_tensors[i].MutableFormat() = src_tensors[i].GetFormat();
    dst_tensors[i].MutableOriginShape() = src_tensors[i].GetOriginShape();
    dst_tensors[i].MutableStorageShape() = src_tensors[i].GetStorageShape();
    dst_tensors[i].SetDataType(src_tensors[i].GetDataType());
    dst_tensors[i].SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
    GE_ASSERT_SUCCESS(TensorTransUtils::AllocDeviceMemory(
        allocator, src_tensors[i].GetSize(), dst_tensors[i], inputs_memblocks[i], aligned_size));

    const size_t src_size = src_tensors[i].GetSize();
    const void *host_addr = src_tensors[i].GetAddr();
    if (src_size <= 0U) {
      continue;
    }

    if (enable_input_batch_cpy) {
      // rts 底层需要 void*, 不会修改，所以 const_cast 是安全的
      MemcpyParam memcpy_param {dst_tensors[i].GetAddr(), aligned_size, const_cast<void *>(host_addr), src_size, attr_idx++}; // NOLINT(*)
      AddMemcpyBatchParam(memcpy_param, memcpy_batch_param);
    } else {
      GE_ASSERT_RT_OK(rtMemcpy(dst_tensors[i].GetAddr(), aligned_size, host_addr, src_size,
                               RT_MEMCPY_HOST_TO_DEVICE));
      GELOGD("Call rtMemcpy success, rt_tensor %p, dst_aligned_size %zu, host_addr %p, src_size %zu",
             dst_tensors[i].GetAddr(), aligned_size, host_addr, src_size);
    }
  }

  GE_ASSERT_SUCCESS(TryBatchMemcpy(memcpy_batch_param));
  return SUCCESS;
}

Status TensorTransUtils::TransTensorToGertTensor(const Tensor &tensor, gert::Tensor &rt_tensor) {
  static std::unordered_map<Placement, gert::TensorPlacement> kPlacementToRtPlacement = {
      {Placement::kPlacementDevice, gert::TensorPlacement::kOnDeviceHbm},
      {Placement::kPlacementHost, gert::TensorPlacement::kOnHost, },
      {Placement::kPlacementEnd, gert::TensorPlacement::kTensorPlacementEnd}
  };
  auto tensor_desc = tensor.GetTensorDesc();
  auto origin_shape = ContructRtShapeFromShape(tensor_desc.GetOriginShape());
  auto storage_shape = ContructRtShapeFromShape(tensor_desc.GetShape());
  rt_tensor.MutableOriginShape() = origin_shape;
  rt_tensor.MutableStorageShape() = storage_shape;
  rt_tensor.MutableFormat() =
      gert::StorageFormat(tensor_desc.GetOriginFormat(), tensor_desc.GetFormat(), gert::ExpandDimsType());
  rt_tensor.SetDataType(tensor_desc.GetDataType());
  GE_ASSERT_TRUE(tensor_desc.GetPlacement() <= Placement::kPlacementEnd);
  rt_tensor.SetPlacement(kPlacementToRtPlacement[tensor_desc.GetPlacement()]);

  GELOGD("Transed ge_tensor %s to rt_tensor %s",
         GeTensorDescToString(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc)).c_str(),
         RtTensorDescToString(rt_tensor).c_str());
  return SUCCESS;
}
Status TensorTransUtils::HostTensorToDeviceGertTensor(ge::Allocator *allocator, const void *src_tensor_addr, uint64_t src_tensor_length,
                                                      gert::Tensor &dst_tensor, MemBlock *&mem_block_to_keep) {
  size_t dst_aligned_size = 0;
  GE_ASSERT_SUCCESS(AllocDeviceMemory(allocator, src_tensor_length, dst_tensor, mem_block_to_keep, dst_aligned_size));

  if (src_tensor_length <= 0U) {
    return SUCCESS;
  }
  GE_ASSERT_RT_OK(rtMemcpy(dst_tensor.GetAddr(), dst_aligned_size, src_tensor_addr, src_tensor_length, RT_MEMCPY_HOST_TO_DEVICE));
  GELOGD("Call rtMemcpy success, dst_tensor %p, dst_aligned_size %zu, src_tensor_addr %p, src_tensor_length %zu",
         dst_tensor.GetAddr(), dst_aligned_size, src_tensor_addr, src_tensor_length);
  return SUCCESS;
}

GeTensor TensorTransUtils::TransRtTensorToGeTensor(const gert::Tensor &input){
  static std::unordered_map<gert::TensorPlacement, Placement> kRtPlacementToPlacement = {
      {gert::TensorPlacement::kOnDeviceHbm, Placement::kPlacementDevice},
      {gert::TensorPlacement::kOnDeviceP2p, Placement::kPlacementDevice},
      {gert::TensorPlacement::kOnHost, Placement::kPlacementHost},
      {gert::TensorPlacement::kFollowing, Placement::kPlacementHost},
      {gert::TensorPlacement::kTensorPlacementEnd, Placement::kPlacementEnd}
  };
  auto shape = ContructGeShapeFromRtShape(input.GetShape().GetStorageShape());
  auto origin_shape = ContructGeShapeFromRtShape(input.GetShape().GetOriginShape());

  GeTensorDesc tensor_desc(shape, input.GetFormat().GetStorageFormat(), input.GetDataType());
  tensor_desc.SetOriginShape(origin_shape);
  tensor_desc.SetOriginFormat(input.GetFormat().GetOriginFormat());
  tensor_desc.SetOriginDataType(input.GetDataType());
  GE_ASSERT_TRUE(input.GetPlacement() <= gert::TensorPlacement::kTensorPlacementEnd);
  tensor_desc.SetPlacement(kRtPlacementToPlacement[input.GetPlacement()]);
  GELOGD("Transed rt_tensor %s to ge_tensor %s", RtTensorDescToString(input).c_str(),
         GeTensorDescToString(tensor_desc).c_str());
  return GeTensor(tensor_desc);
}

Status TensorTransUtils::TransRtTensorToGeTensor(const gert::Tensor &src, GeTensor &dst) {
  static std::unordered_map<gert::TensorPlacement, Placement> kRtPlacementToPlacement = {
      {gert::TensorPlacement::kOnDeviceHbm, Placement::kPlacementDevice},
      {gert::TensorPlacement::kOnDeviceP2p, Placement::kPlacementDevice},
      {gert::TensorPlacement::kOnHost, Placement::kPlacementHost},
      {gert::TensorPlacement::kFollowing, Placement::kPlacementHost},
      {gert::TensorPlacement::kTensorPlacementEnd, Placement::kPlacementEnd}
  };
  auto shape = ContructGeShapeFromRtShape(src.GetShape().GetStorageShape());
  auto origin_shape = ContructGeShapeFromRtShape(src.GetShape().GetOriginShape());
  dst.ClearData();
  auto &tensor_desc = dst.MutableTensorDesc();
  tensor_desc.SetShape(shape);
  tensor_desc.SetFormat(src.GetFormat().GetStorageFormat());
  tensor_desc.SetDataType(src.GetDataType());
  tensor_desc.SetOriginShape(origin_shape);
  tensor_desc.SetOriginFormat(src.GetFormat().GetOriginFormat());
  tensor_desc.SetOriginDataType(src.GetDataType());
  GE_ASSERT_TRUE(src.GetPlacement() <= gert::TensorPlacement::kTensorPlacementEnd);
  tensor_desc.SetPlacement(kRtPlacementToPlacement[src.GetPlacement()]);
  GELOGD("Transed rt_tensor %s to ge_tensor %s", RtTensorDescToString(src).c_str(),
         GeTensorDescToString(tensor_desc).c_str());
  return SUCCESS;
}

Status TensorTransUtils::TransRtTensorToTensor(const std::vector<gert::Tensor> &srcs, std::vector<Tensor> &dsts,
                                               bool with_value) {
  dsts.resize(srcs.size());
  for (size_t i = 0U; i < srcs.size(); ++i) {
    auto &rt_tensor = srcs[i];
    GELOGD("Transed rt_tensor %s to ge_tensor", RtTensorDescToString(rt_tensor).c_str());
    auto shape = ContructGeShapeFromRtShape(rt_tensor.GetShape().GetStorageShape());
    auto ge_tensor = TransRtTensorToGeTensor(rt_tensor);
    if (with_value) {
      if (!shape.IsEmptyTensor()) {
        int64_t output_size = -1;
        GE_ASSERT_SUCCESS((TensorUtils::CalcTensorMemSize(shape, rt_tensor.GetFormat().GetStorageFormat(),
                                                          rt_tensor.GetDataType(), output_size)));
        GE_ASSERT_TRUE(output_size > 0L);
        const auto aligned_ptr = MakeShared<AlignedPtr>(output_size, kValAlignment);
        GE_CHECK_NOTNULL(aligned_ptr);
        auto data_buf = aligned_ptr->MutableGet();
        GE_CHECK_NOTNULL(data_buf);
        GE_CHK_RT_RET(rtMemcpy(data_buf, static_cast<uint64_t>(output_size), rt_tensor.GetAddr(),
                               static_cast<uint64_t>(output_size), RT_MEMCPY_DEVICE_TO_HOST));
        ge_tensor.SetData(aligned_ptr, static_cast<size_t>(output_size));
      } else {
        GE_ASSERT_GRAPH_SUCCESS(ge_tensor.SetData(nullptr, 0));
      }
      ge_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);
    } else {
      ge_tensor.ClearData();
      ge_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementEnd);
    }
    dsts[i] = TensorAdapter::AsTensor(ge_tensor);
  }
  return SUCCESS;
}

// gert::Tensor -> GeTensor
Status TensorTransUtils::GertTensor2GeTensor(const gert::Tensor &gert_tensor, GeTensor &ge_tensor) {
  ge_tensor.SetTensorDesc(GetInnerTensorDescFromGertTensor(gert_tensor));
  auto tensor_data_holder = ge::MakeShared<gert::TensorData>();
  GE_ASSERT_NOTNULL(tensor_data_holder);
  GE_ASSERT_SUCCESS(tensor_data_holder->ShareFrom(gert_tensor.GetTensorData()));
  const auto deleter = [tensor_data_holder](const uint8_t *data) {
    (void) data;
    tensor_data_holder->Free();
  };
  const uint8_t *const addr = ge::PtrToPtr<void, uint8_t>(gert_tensor.GetAddr());
  GE_ASSERT_GRAPH_SUCCESS(
      ge_tensor.SetData(const_cast<uint8_t *>(addr), gert_tensor.GetSize(), deleter));
  return SUCCESS;
}

Status TensorTransUtils::GertTensors2GeTensors(const std::vector<gert::Tensor> &gert_tensors,
  std::vector<GeTensor> &ge_tensors) {
  GE_ASSERT_TRUE(ge_tensors.empty());
  ge_tensors.reserve(gert_tensors.size());
  for (const auto &gert_tensor : gert_tensors) {
    GeTensor ge_tensor;
    GE_ASSERT_SUCCESS(GertTensor2GeTensor(gert_tensor, ge_tensor));
    (void)ge_tensors.emplace_back(std::move(ge_tensor));
  }
  return SUCCESS;
}

// gert::Tensor -> Tensor
Status TensorTransUtils::GertTensor2Tensor(const gert::Tensor &gert_tensor, Tensor &ge_tensor) {
  (void)ge_tensor.SetTensorDesc(GetTensorDescFromGertTensor(gert_tensor));
  auto output_holder = ge::MakeShared<gert::TensorData>();
  GE_ASSERT_NOTNULL(output_holder);
  GE_ASSERT_SUCCESS(output_holder->ShareFrom(gert_tensor.GetTensorData()));
  const auto deleter = [output_holder](const uint8_t *const data) {
    (void) data;
    output_holder->Free();
  };
  const uint8_t *const addr = ge::PtrToPtr<void, uint8_t>(gert_tensor.GetAddr());
  GE_ASSERT_GRAPH_SUCCESS(
      ge_tensor.SetData(const_cast<uint8_t *>(addr), gert_tensor.GetSize(), deleter));
  return SUCCESS;
}

Status TensorTransUtils::GertTensors2Tensors(const std::vector<gert::Tensor> &gert_tensors,
  std::vector<Tensor> &ge_tensors) {
  ge_tensors.resize(gert_tensors.size());
  for (size_t i = 0U; i < gert_tensors.size(); ++i) {
    GE_ASSERT_SUCCESS(GertTensor2Tensor(gert_tensors[i], ge_tensors[i]));
  }
  return SUCCESS;
}

// GeTensor -> gert::Tensor
Status TensorTransUtils::GeTensor2GertTensor(const GeTensor &ge_tensor, gert::Tensor &gert_tensor) {
  const GeTensorDesc &tensor_desc = ge_tensor.GetTensorDesc();
  if (tensor_desc.IsOriginShapeInitialized()) {
    gert_tensor.MutableOriginShape() =
      GetGertShapeFromDims(tensor_desc.GetOriginShape().GetMutableDims());
  } else {
    gert_tensor.MutableOriginShape() = GetGertShapeFromDims(tensor_desc.GetShape().GetMutableDims());
  }
  gert_tensor.MutableStorageShape() = GetGertShapeFromDims(tensor_desc.GetShape().GetMutableDims());
  gert_tensor.SetDataType(tensor_desc.GetDataType());
  gert_tensor.SetOriginFormat(tensor_desc.GetOriginFormat());
  gert_tensor.SetStorageFormat(tensor_desc.GetFormat());

  const auto placement = tensor_desc.GetPlacement();
  const auto rt_placement =
      placement == ge::kPlacementHost ? gert::TensorPlacement::kOnHost : gert::TensorPlacement::kOnDeviceHbm;
  gert_tensor.SetPlacement(rt_placement);

  auto tensor_copy = ge::MakeShared<ge::GeTensor>();
  GE_ASSERT_NOTNULL(tensor_copy);
  TensorUtils::ShareTensor(ge_tensor, *tensor_copy);

  auto tensor_wrapper = new (std::nothrow) TensorWrapper(tensor_copy);
  GE_ASSERT_NOTNULL(tensor_wrapper);
  GE_DISMISSABLE_GUARD(free_if_failed, [tensor_wrapper] () { delete tensor_wrapper;});
  GE_ASSERT_GRAPH_SUCCESS(gert_tensor.MutableTensorData().SetAddr(reinterpret_cast<void *>(tensor_wrapper),
    &TensorWrapper<GeTensor>::Manager));
  gert_tensor.MutableTensorData().SetSize(ge_tensor.GetData().size());
  GE_DISMISS_GUARD(free_if_failed);
  return SUCCESS;
}

Status TensorTransUtils::GeTensors2GertTensors(const std::vector<GeTensor> &ge_tensors,
  std::vector<gert::Tensor> &gert_tensors) {
  gert_tensors.resize(ge_tensors.size());
  for (size_t i = 0U; i < ge_tensors.size(); ++i) {
    GE_ASSERT_SUCCESS(GeTensor2GertTensor(ge_tensors[i], gert_tensors[i]));
  }
  return SUCCESS;
}

// Tensor -> gert::Tensor
Status TensorTransUtils::Tensor2GertTensor(const Tensor &ge_tensor, gert::Tensor &gert_tensor) {
  return GeTensor2GertTensor(ge::TensorAdapter::AsGeTensor(ge_tensor), gert_tensor);
}

Status TensorTransUtils::Tensors2GertTensors(const std::vector<Tensor> &ge_tensors,
  std::vector<gert::Tensor> &gert_tensors) {
  gert_tensors.resize(ge_tensors.size());
  for (size_t i = 0U; i < ge_tensors.size(); ++i) {
    GE_ASSERT_SUCCESS(Tensor2GertTensor(ge_tensors[i], gert_tensors[i]));
  }
  return SUCCESS;
}

// Tensor -> gert::Tensor
Status TensorTransUtils::AsTensorView(const Tensor &ge_tensor, gert::Tensor &tensor_view) {
  tensor_view.MutableOriginShape() = GetGertShapeFromTensor(ge_tensor);
  tensor_view.MutableStorageShape() = tensor_view.GetOriginShape();
  tensor_view.SetDataType(ge_tensor.GetDataType());
  tensor_view.SetOriginFormat(ge_tensor.GetOriginFormat());
  tensor_view.SetStorageFormat(ge_tensor.GetFormat());

  const auto placement = ge_tensor.GetPlacement();
  const auto rt_placement =
      placement == ge::kPlacementHost ? gert::TensorPlacement::kOnHost : gert::TensorPlacement::kOnDeviceHbm;
  tensor_view.SetPlacement(rt_placement);
  GE_ASSERT_GRAPH_SUCCESS(tensor_view.MutableTensorData().SetAddr(ge_tensor.GetData(), nullptr));
  tensor_view.MutableTensorData().SetSize(ge_tensor.GetSize());
  return SUCCESS;
}

Status TensorTransUtils::AsTensorsView(const std::vector<Tensor> &ge_tensors,
  std::vector<gert::Tensor> &tensors_view) {
  tensors_view.resize(ge_tensors.size());
  for (size_t i = 0U; i < ge_tensors.size(); ++i) {
    GE_ASSERT_SUCCESS(AsTensorView(ge_tensors[i], tensors_view[i]));
  }
  return SUCCESS;
}

Status TensorTransUtils::TransGertTensorToHost(const gert::Tensor &src_tensor, gert::Tensor &dst_tensor) {
  // shape, format, data type
  dst_tensor.MutableFormat() = src_tensor.GetFormat();
  dst_tensor.SetDataType(src_tensor.GetDataType());
  dst_tensor.MutableOriginShape() = src_tensor.GetOriginShape();
  dst_tensor.MutableStorageShape() = src_tensor.GetStorageShape();
  dst_tensor.MutableTensorData().SetPlacement(gert::TensorPlacement::kOnHost);

  const auto shape = ContructGeShapeFromRtShape(src_tensor.GetShape().GetStorageShape());
  int64_t output_size = -1;
  if (src_tensor.GetDataType() == DT_STRING) {
    output_size = static_cast<int64_t>(src_tensor.GetSize());
  } else {
    GE_ASSERT_SUCCESS((TensorUtils::CalcTensorMemSize(shape, src_tensor.GetFormat().GetStorageFormat(),
      src_tensor.GetDataType(), output_size)));
  }
  GE_CHECK_GE(output_size, 0L);
  if (output_size > 0L) {
    auto aligned_ptr = MakeShared<AlignedPtr>(output_size, kValAlignment);
    GE_CHECK_NOTNULL(aligned_ptr);
    auto data_buf = aligned_ptr->MutableGet();
    GE_CHECK_NOTNULL(data_buf);
    GE_CHK_RT_RET(rtMemcpy(data_buf, static_cast<uint64_t>(output_size), src_tensor.GetAddr(),
                           static_cast<uint64_t>(output_size), RT_MEMCPY_DEVICE_TO_HOST));

    // 创建 GeTensor 来持有数据，并使用 TensorWrapper 管理生命周期
    auto ge_tensor = ge::MakeShared<GeTensor>();
    GE_ASSERT_NOTNULL(ge_tensor);
    ge_tensor->SetData(aligned_ptr, static_cast<size_t>(output_size));

    auto tensor_wrapper = new (std::nothrow) TensorWrapper<GeTensor>(ge_tensor);
    GE_ASSERT_NOTNULL(tensor_wrapper);
    GE_DISMISSABLE_GUARD(free_if_failed, [tensor_wrapper]() { delete tensor_wrapper; });
    GE_ASSERT_GRAPH_SUCCESS(dst_tensor.MutableTensorData().SetAddr(reinterpret_cast<void *>(tensor_wrapper),
      &TensorWrapper<GeTensor>::Manager));
    dst_tensor.MutableTensorData().SetSize(static_cast<size_t>(output_size));
    GE_DISMISS_GUARD(free_if_failed);
  } else {
    dst_tensor.MutableTensorData().SetAddr(nullptr, nullptr);
    dst_tensor.MutableTensorData().SetSize(0);
  }
  return SUCCESS;
}

Status TensorTransUtils::TransGertTensorsToHost(const std::vector<gert::Tensor> &device_tensors,
    std::vector<gert::Tensor> &host_tensors) {
  host_tensors.reserve(device_tensors.size());
  for (const auto &src : device_tensors) {
    gert::Tensor dst;
    GE_ASSERT_SUCCESS(TransGertTensorToHost(src, dst));
    host_tensors.emplace_back(std::move(dst));
  }
  return SUCCESS;
}

std::vector<gert::Tensor> TensorTransUtils::ShareFromGertTenosrs(const std::vector<gert::Tensor> &gert_tensors) {
  std::vector<gert::Tensor> ret;
  ret.reserve(gert_tensors.size());
  for (const auto &tensor : gert_tensors) {
    gert::Tensor ret_tensor(tensor.GetShape(), tensor.GetFormat(), tensor.GetDataType());
    (void)ret_tensor.MutableTensorData().ShareFrom(tensor.GetTensorData());
    (void)ret.emplace_back(std::move(ret_tensor));
  }
  return ret;
}

void TensorTransUtils::AddMemcpyBatchParam(const MemcpyParam &param, MemcpyBatchParam &memcpy_batch_params) {
  GELOGD("Prepare data for batch memcpy, idx: %zu", param.idx);
  rtMemcpyBatchAttr attr;
  // 仅支持H2D
  attr.srcLoc.type = RT_MEMORY_LOC_HOST;
  attr.dstLoc.type = RT_MEMORY_LOC_DEVICE;
  attr.dstLoc.id = static_cast<uint32_t>(memcpy_batch_params.device_id);

  (void)memcpy_batch_params.dsts.emplace_back(param.dst);
  (void)memcpy_batch_params.dst_aligned_sizes.emplace_back(param.dst_aligned_size);
  (void)memcpy_batch_params.srcs.emplace_back(param.src);
  (void)memcpy_batch_params.src_sizes.emplace_back(param.src_size);
  (void)memcpy_batch_params.attrs.emplace_back(attr);
  (void)memcpy_batch_params.attr_idxs.emplace_back(param.idx);
}

Status TensorTransUtils::TryBatchMemcpy(MemcpyBatchParam &args) {
  if (args.dsts.empty()) {
    GELOGI("No need to batch memcpy");
    return SUCCESS;
  }
  if (args.dsts.size() == 1) {
    GELOGW("The switch of input_batch_cpy is open but only one input remains, not enable batch memcpy");
    GE_ASSERT_TRUE(args.dst_aligned_sizes.size() == 1);
    return static_cast<Status>(rtMemcpy(args.dsts[0],args.dst_aligned_sizes[0], args.srcs[0], args.src_sizes[0], RT_MEMCPY_HOST_TO_DEVICE));
  }
  size_t fail_idx = std::numeric_limits<size_t>::max();
  const rtError_t ret =
      rtsMemcpyBatch(const_cast<void **>(args.dsts.data()), const_cast<void **>(args.srcs.data()),
                     args.src_sizes.data(), args.srcs.size(), args.attrs.data(),
                     args.attr_idxs.data(), args.attrs.size(), &fail_idx);
  if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
    GELOGW("Batch memcpy not supported, ret=%d, fallback to individual memcpy.", ret);
    for (size_t i = 0; i < args.srcs.size(); ++i) {
      GE_ASSERT_RT_OK(rtMemcpy(args.dsts[i], args.dst_aligned_sizes[i], args.srcs[i],
        args.src_sizes[i], RT_MEMCPY_HOST_TO_DEVICE));
    }
    GELOGI("Fallback individual memcpy completed successfully for %zu items", args.dsts.size());
    return SUCCESS;
  }

  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Batch memcpy failed, ret=%d, failed index=%zu", ret, fail_idx);
    return RT_FAILED;
  }

  GELOGI("Batch memcpy success for %zu items", args.dsts.size());
  return SUCCESS;
}

Status TensorTransUtils::FillRtTensorDesc(const Tensor &src_tensor, gert::Tensor &dst_tensor) {
  static const std::unordered_map<Placement, gert::TensorPlacement> kPlacementToRtPlacement = {
      {Placement::kPlacementDevice, gert::TensorPlacement::kOnDeviceHbm},
      {Placement::kPlacementHost, gert::TensorPlacement::kOnHost},
      {Placement::kPlacementEnd, gert::TensorPlacement::kTensorPlacementEnd}
  };

  auto tensor_desc = src_tensor.GetTensorDesc();
  auto origin_shape = ContructRtShapeFromShape(tensor_desc.GetOriginShape());
  auto storage_shape = ContructRtShapeFromShape(tensor_desc.GetShape());
  dst_tensor.MutableOriginShape() = origin_shape;
  dst_tensor.MutableStorageShape() = storage_shape;
  dst_tensor.MutableFormat() =
      gert::StorageFormat(tensor_desc.GetOriginFormat(), tensor_desc.GetFormat(), gert::ExpandDimsType());
  dst_tensor.SetDataType(tensor_desc.GetDataType());
  GE_ASSERT_TRUE(tensor_desc.GetPlacement() <= Placement::kPlacementEnd);
  dst_tensor.SetPlacement(kPlacementToRtPlacement.at(tensor_desc.GetPlacement()));
  GELOGD("FillRtTensorDesc src_tensor %s dst_tensor %s",
         GeTensorDescToString(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc)).c_str(),
         RtTensorDescToString(dst_tensor).c_str());
  return SUCCESS;
}

Status TensorTransUtils::AllocDeviceMemory(ge::Allocator *allocator, uint64_t src_tensor_length, gert::Tensor &dst_tensor,
                                MemBlock *&mem_block_to_keep, size_t &dst_aligned_size) {
  GE_CHECK_LE(src_tensor_length, std::numeric_limits<uint64_t>::max() - (kMemAlignSize * 2U));
  dst_aligned_size = ((src_tensor_length + (kMemAlignSize * 2U) - 1U) / kMemAlignSize) * kMemAlignSize;
  auto mem_block = allocator->Malloc(dst_aligned_size);
  GE_ASSERT_NOTNULL(mem_block);
  GE_ASSERT_NOTNULL(mem_block->GetAddr(), "malloc failed, tensor size=%zu", dst_aligned_size);
  mem_block_to_keep = mem_block;

  dst_tensor.SetData(gert::TensorData{mem_block->GetAddr(), nullptr, mem_block->GetSize(), gert::kOnDeviceHbm});
  return SUCCESS;
}
} // ge
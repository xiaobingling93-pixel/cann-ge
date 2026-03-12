/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_TENSOR_H_
#define ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_TENSOR_H_

#include <vector>
#include <tuple>
#include "runtime/mem.h"
#include "tensor_engine/fusion_types.h"

namespace te {
static constexpr size_t K_TBE_RT_MEMORY_UB = (0x1U << 15U);

class TbeOpTensor {
public:
    TbeOpTensor()
    {
        sub_format_ = 0;
        stype_ = ATTR_SHAPE_TUPLE;
        is_const_ = false;
        L1_workspace_size_ = -1;
        L1_valid_size_ = -1;
        L1_fusion_type_ = -1;
        addr_offset_ = 0;
        split_index_ = 0;
        addr_type_ = 0;
        use_L1_workspace_ = 0;
        L1_addr_flag_ = -1;
        ddr_base_prop_ = DdrBaseType::WORKSPACE;
    }

    ~TbeOpTensor()
    {
    }

    TbeOpTensor(const std::string &name,
                const std::vector<int64_t> &shape,
                const std::string &dtype,
                const std::string &format,
                ATTR_SHAPETYPE stype = ATTR_SHAPE_TUPLE)
        : name_(name),
          shape_(shape),
          shapeRange_(std::tuple<bool, std::vector<std::pair<int64_t, int64_t>>>{false, {}}),
          format_(format),
          sub_format_(0),
          originShapeRange_(std::tuple<bool, std::vector<std::pair<int64_t, int64_t>>>{false, {}}),
          addr_type_(0),
          use_L1_workspace_(0),
          L1_addr_flag_(-1),
          dtype_(dtype),
          stype_(stype),
          L1_fusion_type_(-1),
          addr_offset_(0),
          split_index_(0),
          L1_workspace_size_(-1),
          L1_valid_size_(-1),
          is_first_layer_(std::tuple<bool, bool>(false, false)),
          ddr_base_prop_(DdrBaseType::WORKSPACE)
    {
        is_const_ = false;
    }

    void GetName(std::string& name) const
    {
        name = name_;
    }

    const std::string& GetName() const
    {
        return name_;
    }

    void SetShape(const std::vector<int64_t> &shape)
    {
        shape_.assign(shape.begin(), shape.end());
    }

    void GetShape(std::vector<int64_t>& shape) const
    {
        shape = shape_;
    }

    const std::vector<int64_t>& GetShape() const
    {
        return shape_;
    }

    void SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &shapeRange)
    {
        shapeRange_ = std::make_tuple(true, shapeRange);
    }

    bool GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &shapeRange) const
    {
        if (!std::get<0>(shapeRange_)) {
            return false;
        }

        shapeRange = std::get<1>(shapeRange_);
        return true;
    }

    const std::vector<std::pair<int64_t, int64_t>>& GetShapeRange() const
    {
        return std::get<1>(shapeRange_);
    }

    void SetOriginShape(const std::vector<int64_t> &originShape)
    {
        origin_shape_.assign(originShape.begin(), originShape.end());
    }

    void GetOriginShape(std::vector<int64_t>& originShape) const
    {
        originShape = origin_shape_;
    }

    const std::vector<int64_t>& GetOriginShape() const
    {
        return origin_shape_;
    }

    void SetOriginShapeRange(const std::vector<std::pair<int64_t, int64_t>> &shapeRange)
    {
        originShapeRange_ = std::make_tuple(true, shapeRange);
    }

    bool GetOriginShapeRange(std::vector<std::pair<int64_t, int64_t>> &shapeRange) const
    {
        if (!std::get<0>(originShapeRange_)) {
            return false;
        }

        shapeRange = std::get<1>(originShapeRange_);
        return true;
    }

    const std::vector<std::pair<int64_t, int64_t>>& GetOriginShapeRange() const
    {
        return std::get<1>(originShapeRange_);
    }

    void SetType(const std::string &dtype)
    {
        dtype_ = dtype;
    }

    void GetType(std::string &dtype) const
    {
        dtype = dtype_;
    }

    const std::string& GetType() const
    {
        return dtype_;
    }

    void GetShapeType(ATTR_SHAPETYPE& stype) const
    {
        stype = stype_;
    }

    ATTR_SHAPETYPE GetShapeType() const
    {
        return stype_;
    }

    void SetShapeType(ATTR_SHAPETYPE stype)
    {
        stype_ = stype;
    }

    void SetFormat(const std::string &format)
    {
        format_ = format;
    }

    void GetFormat(std::string& format) const
    {
        format = format_;
    }

    const std::string& GetFormat() const
    {
        return format_;
    }

    void SetSubFormat(const int32_t subFormat)
    {
        sub_format_ = subFormat;
    }

    void GetSubFormat(int32_t& subFormat) const
    {
        subFormat = sub_format_;
    }

    int32_t GetSubFormat() const
    {
        return sub_format_;
    }

    void SetOriginFormat(const std::string &originFormat)
    {
        origin_format_ = originFormat;
    }

    void GetOriginFormat(std::string& originFormat) const
    {
        originFormat = origin_format_;
    }

    const std::string& GetOriginFormat() const
    {
        return origin_format_;
    }

    void SetConstValue(const TbeAttrValue& value)
    {
        is_const_ = true;
        const_value_ = value;
    }

    void GetConstFlag(bool& bl) const
    {
        bl = is_const_;
    }

    bool HasConstValue() const
    {
        return is_const_;
    }

    void GetConstValue(TbeAttrValue& value) const
    {
        value = const_value_;
    }

    const TbeAttrValue& GetConstValue() const
    {
        return const_value_;
    }

    void SetConstValueRange(const TbeAttrValue &valueRange)
    {
        is_const_value_range_ = true;
        const_value_range_ = valueRange;
    }

    void GetConstValueRangeFlag(bool &isConstValueRange) const
    {
        isConstValueRange = is_const_value_range_;
    }

    bool IsConstValueRange() const
    {
        return is_const_value_range_;
    }

    void GetConstValueRange(TbeAttrValue &valueRange) const
    {
        valueRange = const_value_range_;
    }

    const TbeAttrValue& GetConstValueRange() const
    {
        return const_value_range_;
    }

    void SetConstValueNone(const bool isValueNone)
    {
        is_const_value_none_ = isValueNone;
    }

    void IsConstValueNone(bool &isConstValueNone) const
    {
        isConstValueNone = is_const_value_none_;
    }

    bool IsConstValueNone() const
    {
        return is_const_value_none_;
    }

    void GetAddrType(size_t& value) const
    {
        value = addr_type_;
    }

    size_t GetAddrType() const
    {
        return addr_type_;
    }

    void SetAddrType(const size_t value)
    {
        switch (value) {
            case RT_MEMORY_HBM:
                addr_type_ = TBE_MEMORY_DDR;
                break;
            case RT_MEMORY_L1:
                addr_type_ = TBE_MEMORY_L1;
                break;
            case RT_MEMORY_L2:
                addr_type_ = TBE_MEMORY_L2;
                break;
            case K_TBE_RT_MEMORY_UB:
                addr_type_ = TBE_MEMORY_UB;
                break;
            default:
                addr_type_ = TBE_MEMORY_DDR;
                break;
        }
    }

    void SetValidShape(const std::vector<int64_t> &shape)
    {
        valid_shape_.assign(shape.begin(), shape.end());
    }

    void GetValidShape(std::vector<int64_t>& shape) const
    {
        shape = valid_shape_;
    }

    const std::vector<int64_t>& GetValidShape() const
    {
        return valid_shape_;
    }

    void SetSgtSliceShape(const std::vector<int64_t> &shape)
    {
        sgt_slice_shape_.assign(shape.begin(), shape.end());
    }

    void GetSgtSliceShape(std::vector<int64_t>& shape) const
    {
        shape = sgt_slice_shape_;
    }

    const std::vector<int64_t>& GetSgtSliceShape() const
    {
        return sgt_slice_shape_;
    }

    void SetSliceOffset(const std::vector<int64_t> &offset)
    {
        slice_offset_.assign(offset.begin(), offset.end());
    }

    void GetSliceOffset(std::vector<int64_t>& offset) const
    {
        offset = slice_offset_;
    }

    const std::vector<int64_t>& GetSliceOffset() const
    {
        return slice_offset_;
    }

    void SetL1WorkspaceFlag(const size_t value)
    {
        use_L1_workspace_ = value;
    }

    void GetL1WorkspaceFlag(size_t &value) const
    {
        value = use_L1_workspace_;
    }

    size_t GetL1WorkspaceFlag() const
    {
        return use_L1_workspace_;
    }

    void SetL1AddrFlag(const int64_t value)
    {
        L1_addr_flag_ = value;
    }

    void GetL1AddrFlag(int64_t &value) const
    {
        value = L1_addr_flag_;
    }

    int64_t GetL1AddrFlag() const
    {
        return L1_addr_flag_;
    }

    void GetL1FusionType(int32_t& l1Type) const
    {
        l1Type = L1_fusion_type_;
    }

    int32_t GetL1FusionType() const
    {
        return L1_fusion_type_;
    }

    void SetL1FusionType(const int32_t l1Type)
    {
        L1_fusion_type_ = l1Type;
    }

    void SetAddrOffset(const int64_t offset)
    {
        addr_offset_ = offset;
    }

    void GetAddrOffset(int64_t &offset) const
    {
        offset = addr_offset_;
    }

    int64_t GetAddrOffset() const
    {
        return addr_offset_;
    }

    void SetCAxisValue(const int64_t &cAxisValue)
    {
        cAxisValue_ = cAxisValue;
    }

    int64_t GetCAxisValue() const
    {
        return cAxisValue_;
    }

    void GetCAxisValue(int64_t &cAxisValue) const
    {
        cAxisValue = cAxisValue_;
    }

    void SetTotalShape(const std::vector<uint32_t>& shape)
    {
        total_shape_ = shape;
    }

    void GetTotalShape(std::vector<uint32_t>& shape) const
    {
        shape = total_shape_;
    }

    const std::vector<uint32_t>& GetTotalShape() const
    {
        return total_shape_;
    }

    void SetSplitIndex(const uint32_t index)
    {
        split_index_ = index;
    }

    uint32_t GetSplitIndex() const
    {
        return split_index_;
    }

    void GetSplitIndex(uint32_t &index) const
    {
        index = split_index_;
    }

    void SetL1WorkspaceSize(const int64_t l1size)
    {
        L1_workspace_size_ = l1size;
    }

    int64_t GetL1WorkspaceSize() const
    {
        return L1_workspace_size_;
    }

    void GetL1WorkspaceSize(int64_t &l1size) const
    {
        l1size = L1_workspace_size_;
    }

    void SetL1ValidSize(const int64_t l1size)
    {
        L1_valid_size_ = l1size;
    }

    void GetL1ValidSize(int64_t &l1size) const
    {
        l1size = L1_valid_size_;
    }

    int64_t GetL1ValidSize() const
    {
        return L1_valid_size_;
    }

    void SetFirstLayer(const bool isFirstLayer)
    {
        is_first_layer_ = std::make_tuple(true, isFirstLayer);
    }

    bool GetFirstLayer(bool &isFirstLayer) const
    {
        if (!std::get<0>(is_first_layer_)) {
            // this parameter has not been set
            return false;
        }

        isFirstLayer = std::get<1>(is_first_layer_);
        return true;
    }

    bool GetFirstLayer() const
    {
        return std::get<1>(is_first_layer_);
    }

    void SetValueRange(const std::vector<std::pair<int64_t, int64_t>> &valueRange)
    {
        value_range_ = valueRange;
    }

    bool GetValueRange(std::vector<std::pair<int64_t, int64_t>> &valueRange) const
    {
        if (value_range_.empty()) {
            return false;
        }

        valueRange = value_range_;
        return true;
    }

    void SetAtomicType(const std::string &atomicType)
    {
        atomic_type_ = atomicType;
    }

    void GetAtomicType(std::string &atomicType) const
    {
        atomicType = atomic_type_;
    }

    void SetIsInputConst(const int32_t &isConstInput)
    {
        is_input_const_ = isConstInput;
    }

    int32_t GetInputConst() const
    {
        return is_input_const_;
    }

    const std::string& GetAtomicType() const
    {
        return atomic_type_;
    }

    DdrBaseType GetDdrBaseProp() const
    {
        return ddr_base_prop_;
    }

    void SetDdrBaseProp(const DdrBaseType &ddrBaseProp)
    {
        ddr_base_prop_ = ddrBaseProp;
    }

    void SetIsNullOutput(const bool is_null_output)
 	{
 	    is_null_output_ = is_null_output;
 	}
 	  
 	void GetIsNullOutput(bool &is_null_output) const
 	{
 	    is_null_output = is_null_output_;
 	}

    bool operator==(TbeOpTensor& rObject);

private:
    // need to adapt operator== func while adding new variable
    std::string name_;
    // current shape and format
    std::vector<int64_t> shape_;
    std::tuple<bool, std::vector<std::pair<int64_t, int64_t>>> shapeRange_;

    std::string format_;
    int32_t sub_format_;

    // original shape and format
    std::vector<int64_t> origin_shape_;
    std::tuple<bool, std::vector<std::pair<int64_t, int64_t>>> originShapeRange_;
    std::string origin_format_;

    // L1 fusion parameter
    size_t addr_type_;
    std::vector<int64_t> valid_shape_;
    std::vector<int64_t> slice_offset_;
    size_t use_L1_workspace_;
    int64_t L1_addr_flag_;
    std::string dtype_;
    ATTR_SHAPETYPE stype_;
    bool is_const_{false};
    bool is_const_value_none_{false};
    bool is_const_value_range_{false};
    TbeAttrValue const_value_;
    TbeAttrValue const_value_range_;
    int32_t L1_fusion_type_;
    int64_t addr_offset_;
    std::vector<uint32_t> total_shape_;
    uint32_t split_index_;
    int64_t L1_workspace_size_;
    int64_t L1_valid_size_;
    std::tuple<bool, bool> is_first_layer_;
    std::vector<std::pair<int64_t, int64_t>> value_range_;
    std::vector<int64_t> sgt_slice_shape_;
    std::string atomic_type_;  // include add sub mul div, etc..
    int64_t cAxisValue_{-1};
    DdrBaseType ddr_base_prop_;
    int32_t is_input_const_{-1};
    bool is_null_output_{false};
};
}
#endif  // ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_TENSOR_H_
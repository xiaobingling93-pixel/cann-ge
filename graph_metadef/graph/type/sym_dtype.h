/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_GRAPH_SYM_DTYPE_H_
#define METADEF_CXX_GRAPH_SYM_DTYPE_H_

#include "graph/op_desc.h"
#include "graph/utils/type_utils.h"

namespace ge {
class OrderedTensorTypeList {
 public:
  explicit OrderedTensorTypeList(const std::initializer_list<DataType> &initial_types) : initial_types_(initial_types) {
  }
  std::vector<DataType> GetOrderedDtypes() const {
    return initial_types_;
  }
  bool IsDataTypeInRange(ge::DataType data_type) const {
    return std::find(initial_types_.begin(), initial_types_.end(), data_type) != initial_types_.end();
  }
  std::vector<size_t> GetDtypeIndexs(ge::DataType data_type) const {
    std::vector<size_t> indices;
    size_t idx = 0;
    for (auto it = initial_types_.begin(); it != initial_types_.end(); ++it, ++idx) {
      if (*it == data_type) {
        indices.push_back(idx);
      }
    }
    return indices;
  }
  std::string ToString() const {
    std::string s = "[";
    for (const auto &dtype : initial_types_) {
      s += TypeUtils::DataTypeToSerialString(dtype);
      s += ",";
    }
    s += "]";
    return s;
  }
 private:
  std::vector<DataType> initial_types_;
};

class TypeOrTypes {
 public:
  TypeOrTypes() : initialized_(false), is_list_(false) {}
  ~TypeOrTypes() = default;

  bool IsListType() const {
    return is_list_;
  }

  graphStatus GetType(DataType &type) const;
  graphStatus GetTypes(std::vector<DataType> &types) const;

  const DataType &UnsafeGetType() const;
  const std::vector<DataType> &UnsafeGetTypes() const;

  void SetType(const DataType &type);
  void SetTypes(const std::vector<DataType> &types);

  std::string DebugString() const;

 private:
  bool initialized_;
  bool is_list_;
  std::vector<DataType> types_;
};

enum class ExpressionType {
  kSingle,
  kPromote
};

// 用于支持符号推导的数据类型表达式，它表达一个由Sym组成的表达式
class SymDtypeExpression {
 public:
  // 对Sym表达式进行基于op上下文的实际值计算
  virtual graphStatus Eval(const OpDesc &op, TypeOrTypes &type_or_types) const = 0;
  virtual bool IsListType() const = 0;
  virtual ExpressionType Type() const = 0;
  virtual std::vector<size_t> GetIrInputIndexes() const = 0;
  virtual ~SymDtypeExpression() = default;
};

// Sym类型，每个输入或输出对应一个Sym，多个输入或输出可以对应同一个Sym
class SymDtype : public SymDtypeExpression {
 public:
  explicit SymDtype(const std::string &id);
  ~SymDtype() override = default;

  const std::string &Id() const;     // Sym的标识，与DATATYPE中声明时的标识一致
  bool IsListType() const override;  // 返回Sym是否对为ListType类型
  ExpressionType Type() const override;

  graphStatus Eval(const OpDesc &op, TypeOrTypes &type_or_types) const override;

  // 绑定sym对应的IR输入名称，以及IR输入类型。如果未绑定任何输入，在推导时会尝试从属性中获取
  void BindIrInput(const std::string &ir_input, const IrInputType &input_type, size_t input_index);
  // 设置Sym的取值范围或计算方式（DATATYPE声明时调用）
  void BindAllowedDtypes(const TensorType &types);
  void BindAllowedDtypes(const ListTensorType &types);
  void BindAllowedOrderedDtypes(const OrderedTensorTypeList &types);
  void BindExpression(const std::shared_ptr<SymDtypeExpression> &expression);

  bool IsLegacy() const;  // 是否为Legacy的Sym，未通过DATATYPE声明的Sym为Legacy的Sym
  bool IsOrderedList() const {
    return is_ordered_list_;
  }

  const std::string &DebugString() const;

  // 仅返回当前Sym对应的输入实体索引
  std::vector<size_t> GetDirectIrInputIndexes() const;

  // 返回当前Sym涉及的输入实体索引：
  // - 由Sym组成的表达式，返回表达式中所有Sym对应的输入实体索引
  // - 非表达式，返回该Sym对应的输入实体索引
  std::vector<size_t> GetIrInputIndexes() const override;

  TensorType GetTensorType() const {
    return tensor_type_;
  }

  OrderedTensorTypeList GetOrderedTensorTypeList() const {
    return ordered_tensor_type_list_;
  }
 protected:
  graphStatus Eval(const OpDesc &op, DataType &dtype) const;
  graphStatus Eval(const OpDesc &op, std::vector<DataType> &dtypes) const;

  std::string id_;
  bool is_legacy_;  // 是否为Legacy方式的IR，对于Legacy方式的IR，不支持类型推导及类型校验

  bool is_list_;            // Sym是否为ListType类型
  TensorType tensor_type_;  // Sym的取值范围
  bool is_ordered_list_;
  OrderedTensorTypeList ordered_tensor_type_list_;

  struct SymBackend {
    SymBackend(const std::string &input_name, const IrInputType &input_type, size_t input_index)
        : type(input_type), index(input_index), name(input_name) {}
    IrInputType type;
    size_t index;
    std::string name;
    std::string DebugString() const;
  };

  std::vector<SymBackend> ir_inputs_;               // Sym的对应的输入实体，与expression_互斥
  std::shared_ptr<SymDtypeExpression> expression_;  // Sym的计算表达式，与ir_inputs_互斥
};

// 表达类型提升的Sym表达式
class PromotionSymDtypeExpression : public SymDtypeExpression {
 public:
  // 表达类型提升的Sym计算表达，入参syms中的sym进行两两提升，对ListType类型的sym，会继续对sym间对应位置的Dtype进行提升
  explicit PromotionSymDtypeExpression(const std::vector<SymDtype *> &syms);

  graphStatus Eval(const OpDesc &op, TypeOrTypes &type_or_types) const override;
  bool IsListType() const override {
    return std::all_of(syms_.begin(), syms_.end(),
                       [](const SymDtype *sym) { return (sym != nullptr) && sym->IsListType(); });
  }
  ExpressionType Type() const override;
  std::vector<size_t> GetIrInputIndexes() const override;

 private:
  std::vector<SymDtype *> syms_;
};

// 获取输入IR对应的实例Desc的index范围，实例Desc中会去除未传值的Optional输入Desc
graphStatus GetIrInputInstanceDescRange(const OpDescPtr &op,
                                        std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range);

// 获取输入IR对应的全部Desc的index范围，包含未传值的Optional输入Desc
graphStatus GetIrInputRawDescRange(const OpDescPtr &op, std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range);

graphStatus GetIrOutputDescRange(const OpDescPtr &op, std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range);
}  // namespace ge
#endif  // METADEF_CXX_GRAPH_SYM_DTYPE_H_

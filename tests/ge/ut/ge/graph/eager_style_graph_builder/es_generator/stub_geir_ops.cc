/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator_reg.h"
namespace ge {
REG_OP(Add)
    .INPUT(x1, TensorType::ALL())
    .INPUT(x2, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Add);

REG_OP(phony_1i_1o)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(phony_1i_1o);

REG_OP(phony_1i1dyi_1o)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_INPUT(dx, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, ListInt, {10, 10, 10})
    .OP_END_FACTORY_REG(phony_1i1dyi_1o);

REG_OP(phony_1i1opi_1o)
    .INPUT(x, TensorType::ALL())
    .OPTIONAL_INPUT(dx, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(dt, Type, DT_FLOAT)
    .REQUIRED_ATTR(flag, Bool)
    .OP_END_FACTORY_REG(phony_1i1opi_1o);

REG_OP(phony_3opi_1o)
    .OPTIONAL_INPUT(x1, TensorType::ALL())
    .OPTIONAL_INPUT(x2, TensorType::ALL())
    .OPTIONAL_INPUT(x3, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(phony_3opi_1o);

REG_OP(phony_1opi_1o)
    .OPTIONAL_INPUT(x1, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(flag, Bool)
    .OP_END_FACTORY_REG(phony_1opi_1o);

REG_OP(phony_2opi1i_1o)
    .OPTIONAL_INPUT(x1, TensorType::ALL())
    .OPTIONAL_INPUT(x2, TensorType::ALL())
    .INPUT(x3, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(phony_2opi1i_1o);

REG_OP(phony_1opi1i_1o)
    .INPUT(x, TensorType::ALL())
    .OPTIONAL_INPUT(dx, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(dt, Type, DT_FLOAT)
    .REQUIRED_ATTR(flag, Bool)
    .OP_END_FACTORY_REG(phony_1opi1i_1o);

REG_OP(phony_1i_2o)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y1, TensorType::ALL())
    .OUTPUT(y2, TensorType::ALL())
    .OP_END_FACTORY_REG(phony_1i_2o);

REG_OP(phony_1i_1dyo)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy, TensorType::ALL())
    .ATTR(index, ListInt, {10, 10, 10})
    .OP_END_FACTORY_REG(phony_1i_1dyo);

REG_OP(phony_1i1dyi_1dyo)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_INPUT(dx, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy, TensorType::ALL())
    .OP_END_FACTORY_REG(phony_1i1dyi_1dyo);

REG_OP(phony_1i_1o1dyo)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy, TensorType::ALL())
    .OP_END_FACTORY_REG(phony_1i_1o1dyo);

REG_OP(phony_1i_1dyo1o)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, ListInt, {10, 10, 10})
    .OP_END_FACTORY_REG(phony_1i_1dyo1o);

REG_OP(phony_1i_2o1dyo)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y1, TensorType::ALL())
    .OUTPUT(y2, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy, TensorType::ALL())
    .OP_END_FACTORY_REG(phony_1i_2o1dyo);

REG_OP(phony_1i_2dyo)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy1, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy2, TensorType::ALL())
    .ATTR(index, ListInt, {10, 10, 10})
    .OP_END_FACTORY_REG(phony_1i_2dyo);

REG_OP(phony_1i1dyi_3dyo)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_INPUT(dx, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy1, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy2, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy3, TensorType::ALL())
    .OP_END_FACTORY_REG(phony_1i1dyi_3dyo);

REG_OP(phony_1i1dyi_2o2dyo1o)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_INPUT(dx, TensorType::ALL())
    .OUTPUT(y1, TensorType::ALL())
    .OUTPUT(y2, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy1, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy2, TensorType::ALL())
    .OUTPUT(y3, TensorType::ALL())
    .DYNAMIC_OUTPUT(dy3, TensorType::ALL())
    .ATTR(index, ListInt, {10, 10, 10})
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(phony_1i1dyi_2o2dyo1o);

REG_OP(phony_multi_attr)
    .ATTR(li, ListInt, {10, 10, 10})
    .ATTR(f, Float, 0.0)
    .ATTR(s, String, "s")
    .ATTR(b, Bool, true)
    .ATTR(lf, ListFloat, {0.1, 0.2})
    .ATTR(lb, ListBool, {false, true})
    .OP_END_FACTORY_REG(phony_multi_attr);

// Additional Attrs
// 1. VT_DATA_TYPE,
// 2. VT_LIST_DATA_TYPE,
// 3. VT_LIST_LIST_INT
// 4. VT_TENSOR
// 5. VT_LIST_STRING
REG_OP(phony_req_attrs)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(req_data_type, Type)
    .REQUIRED_ATTR(req_list_data_type, ListType)
    .REQUIRED_ATTR(req_list_list_int, ListListInt)
    .REQUIRED_ATTR(req_tensor, Tensor)
    .REQUIRED_ATTR(req_list_string, ListString)
    .OP_END_FACTORY_REG(phony_req_attrs)
REG_OP(phony_opt_attrs)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(opt_data_type, Type, DT_INT64)
    .ATTR(opt_list_data_type, ListType, {DT_FLOAT, DT_DOUBLE})
    .ATTR(opt_list_list_int, ListListInt, {{1,2,3}, {3,2,1}})
    .ATTR(opt_tensor, Tensor, Tensor())
    .ATTR(opt_list_string, ListString, {"test", "test"})
    .OP_END_FACTORY_REG(phony_opt_attrs)

// Op with Subgraphs
REG_OP(phony_If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(phony_If)
REG_OP(phony_Case)
    .INPUT(branch_index, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .DYNAMIC_GRAPH(branches)
    .OP_END_FACTORY_REG(phony_Case)
REG_OP(phony_PartitionedCall)
    .DYNAMIC_INPUT(args, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(f)
    .ATTR(config, String, "")
    .ATTR(config_proto, String, "")
    .ATTR(executor_type, String, "")
    .OP_END_FACTORY_REG(phony_PartitionedCall)

// Op with subgraph and should handle 'cond'
REG_OP(While)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(cond)
    .GRAPH(body)
    .ATTR(parallel_iterations, Int, 10)
    .OP_END_FACTORY_REG(While)

// Op with dynamic and static subgraphs
REG_OP(phony_mix_subgraphs)
    .OPTIONAL_INPUT(opt_input, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(cond)
    .DYNAMIC_GRAPH(branches)
    .OP_END_FACTORY_REG(phony_mix_subgraphs)

// Following are V1 control Ops which should be filtered out
REG_OP(Switch)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .INPUT(pred, TensorType({DT_BOOL}))
    .OUTPUT(output_false, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(output_true, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(Switch)
REG_OP(StreamSwitch) // Fake
    .INPUT(input, TensorType::ALL())
    .OUTPUT(output, TensorType::ALL())
    .OP_END_FACTORY_REG(StreamSwitch)
REG_OP(Merge)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(value_index, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Merge)
REG_OP(StreamMerge) // Fake
    .INPUT(input, TensorType::ALL())
    .OUTPUT(output, TensorType::ALL())
    .OP_END_FACTORY_REG(StreamMerge)
REG_OP(Enter)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .REQUIRED_ATTR(frame_name, String)
    .REQUIRED_ATTR(is_constant, Bool)
    .OP_END_FACTORY_REG(Enter)
REG_OP(Exit)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(Exit)
REG_OP(LoopCond)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LoopCond)
REG_OP(NextIteration)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(NextIteration)

// Const and Variable Op will be filtered out since we already provided related creation functions
REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)
REG_OP(Variable)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Variable);

// Op with duplicat names
REG_OP(phony_dup_name)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_INPUT(dx, TensorType::ALL())
    .DYNAMIC_OUTPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(dx, TensorType::ALL())
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .ATTR(x, Int, 1)
    .ATTR(dx, Int, 2)
    .OP_END_FACTORY_REG(phony_dup_name);

// Invalid Ops
REG_OP(phony_same_name)
    .INPUT(x, TensorType::ALL())
    .ATTR(x, ListInt, {10, 10, 10})
    .OP_END_FACTORY_REG(phony_same_name);
REG_OP(phony_invalid_attr_value)
    .INPUT(x, TensorType::ALL())
    .ATTR(dt, Type, ge::DataType(-1))
    .OP_END_FACTORY_REG(phony_invalid_attr_value);
}  // namespace ge

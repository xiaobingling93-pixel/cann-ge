/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "perf_param_v1.h"
#include <numeric>
#include "base/att_const_values.h"
#include "gen_model_info/parser/tuning_space.h"

namespace att {
namespace {
const uint64_t kLoadThreshold = 25000U;
const Expr kVecheadcost = CreateExpr(37.37f);
const Expr kMTE2Thres = CreateExpr(25000U);
const Expr kMte2PipeHeadNormalK = CreateExpr(32.72f);
const Expr kMte2PipeHeadNormalB = CreateExpr(1575.03f);
const Expr kMte2PipeHeadSmallK = CreateExpr(15.89f);
const Expr kMte2PipeHeadSmallB = CreateExpr(882.09f);
const Expr kMte3PipeHeadcost = CreateExpr(497.36f);
}
Expr PerfParamTableV1::GetMTE2PipeHead(const std::vector<NodeInfo> &node_infos, std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) {
  Expr mte2_pipe_head;
  Expr max_data_size = ge::sym::kSymbolZero;
  for (const auto &node : node_infos) {
    if (node.node_type == kLoad) {
      if (node.inputs.size() != 1) {
        GELOGW("Got node %s input size %zu is not 1.", node.name.c_str(), node.inputs.size());
        continue;
      }
      auto dims = node.inputs[0]->repeat;
      Expr dim_product =
          std::accumulate(dims.begin(), dims.end(), CreateExpr(1), [](const Expr &a, const Expr &b) { return a * b; });
      Expr data_size = dim_product * CreateExpr(node.inputs[0]->data_type_size);
      if (max_data_size != ge::sym::kSymbolZero) {
        max_data_size = ge::sym::Max(data_size, max_data_size);
      } else {
        max_data_size = data_size;
      }
    }
  }
  const auto block_dim = CreateExpr("block_dim");
  Expr small_head_cost = block_dim * kMte2PipeHeadSmallK + kMte2PipeHeadSmallB;
  Expr normal_head_cost = block_dim * kMte2PipeHeadNormalK + kMte2PipeHeadNormalB;
  if (max_data_size.IsConstExpr()) {
    uint64_t datasize;
    max_data_size.GetConstValue(datasize);
    if (datasize >= kLoadThreshold) {
      mte2_pipe_head = normal_head_cost;
    } else {
      mte2_pipe_head = small_head_cost;
    }
  } else {
    GetPerfVar("mte2_pipe_head", mte2_pipe_head, ternary_ops);
    TernaryOp ternary_op = TernaryOp(CondType::K_LT, max_data_size, kMTE2Thres, small_head_cost, normal_head_cost);
    ternary_op.SetVariable(mte2_pipe_head);
    ternary_ops[mte2_pipe_head] = ternary_op;
  }
  return mte2_pipe_head;
}

static const std::string kParamV1Info = R"(
{
    "Abs": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0077, "b": 20.0153},
            "float32tofloat32": {"k": 0.0147, "b": 20.0592}
        }
    },
    "Adds": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 22.0938},
            "float32tofloat32": {"k": 0.0141, "b": 22.0936}
        }
    },
    "Add": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0103, "b": 22.2173},
            "float32tofloat32": {"k": 0.0206, "b": 23.2225}
        }
    },
    "And": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0107, "b": 17.1393},
            "float32tofloat32": {"k": 0.0112, "b": 17.5611}
        }
    },
    "BlockReduceMax": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 18.0042},
            "float32tofloat32": {"k": 0.0198, "b": 16.7401}
        }
    },
    "BlockReduceMin": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 18.0092},
            "float32tofloat32": {"k": 0.0198, "b": 16.7413}
        }
    },
    "Brcb": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0074, "b": 13.0572},
            "float32tofloat32": {"k": 0.0146, "b": 13.0732}
        }
    },
    "Cast": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0, "b": 0},
            "float32tofloat32": {"k": 0, "b": 0},
            "float16tofloat32": {"k": 0.0147, "b": 20.1204},
            "float32tofloat16": {"k": 0.0087, "b": 20.4393},
            "float16touint8": {"k": 0.0083, "b": 19.9408},
            "uint8tofloat16": {"k": 0.0102, "b": 46.512},
            "int64toint32": {"k": 0.0332, "b": 18.8690}
        }
    },
    "Ub2ub": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0076, "b": 11.6372},
            "float32tofloat32": {"k": 0.0152, "b": 11.6372}
        }
    },
    "Copy": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0078, "b": 13.0049},
            "float32tofloat32": {"k": 0.0157, "b": 12.9966}
        }
    },
    "CompareScalarEQ": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9204},
            "float32tofloat32": {"k": 0.0160, "b": 21.9749}
        }
    },
    "CompareScalarGE": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0086, "b": 21.9025},
            "float32tofloat32": {"k": 0.0161, "b": 21.9643}
        }
    },
    "CompareScalarGT": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9200},
            "float32tofloat32": {"k": 0.0160, "b": 21.9712}
        }
    },
    "CompareScalarLE": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9210},
            "float32tofloat32": {"k": 0.0161, "b": 21.9722}
        }
    },
    "CompareScalarNE": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9114},
            "float32tofloat32": {"k": 0.0161, "b": 22.9690}
        }
    },
    "CompareScalarLT": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9381},
            "float32tofloat32": {"k": 0.0161, "b": 22.9860}
        }
    },
    "CompareEQAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0155, "b": 21.0340},
            "float32tofloat32": {"k": 0.0310, "b": 21.0316}
        }
    },
    "CompareEQUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0081, "b": 21.0340},
            "float32tofloat32": {"k": 0.0157, "b": 21.0316}
        }
    },
    "CompareGEAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0155, "b": 21.0279},
            "float32tofloat32": {"k": 0.0310, "b": 21.0309}
        }
    },
    "CompareGEUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0079, "b": 21.0279},
            "float32tofloat32": {"k": 0.0156, "b": 21.0309}
        }
    },
    "CompareGTAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0160, "b": 20.9711},
            "float32tofloat32": {"k": 0.0310, "b": 21.0319}
        }
    },
    "CompareGTUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0114, "b": 20.9711},
            "float32tofloat32": {"k": 0.0157, "b": 21.0319}
        }
    },
    "CompareLEAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0155, "b": 21.9556},
            "float32tofloat32": {"k": 0.0310, "b": 21.0303}
        }
    },
    "CompareLEUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0080, "b": 21.9556},
            "float32tofloat32": {"k": 0.0158, "b": 21.0303}
        }
    },
    "CompareLTAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0155, "b": 21.0297},
            "float32tofloat32": {"k": 0.0316, "b": 20.9940}
        }
    },
    "CompareLTUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0079, "b": 21.0297},
            "float32tofloat32": {"k": 0.0173, "b": 20.9940}
        }
    },
    "CompareNEAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0164, "b": 20.9123},
            "float32tofloat32": {"k": 0.0310, "b": 21.0313}
        }
    },
    "CompareNEUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0115, "b": 20.9123},
            "float32tofloat32": {"k": 0.0156, "b": 21.0313}
        }
    },
    "PowerAllTensor": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.6647, "b": 2516.53},
            "float32tofloat32": {"k": 0.6199, "b": 730.11}
        }
    },
    "PowerWithScalar": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.64965, "b": 2517.18},
            "float32tofloat32": {"k": 0.69805, "b": 735.05}
        }
    },
    "Div": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0460, "b": 29.1233},
            "float32tofloat32": {"k": 0.0454, "b": 29.0892}
        }
    },
    "Duplicate": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0078, "b": 16.9993},
            "float32tofloat32": {"k": 0.0156, "b": 16.9965}
        }
    },
    "Erf": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.6996, "b": 478.4175},
            "float32tofloat32": {"k": 0.6038, "b": 458.2933}
        }
    },
    "Exp": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0311, "b": 28.0144},
            "float32tofloat32": {"k": 0.0307, "b": 28.0376}
        }
    },
    "Gather": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.1873, "b": 17.0248},
            "float32tofloat32": {"k": 0.1875, "b": 15.0000}
        }
    },
    "GatherMask": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0156, "b": 14.0242},
            "float32tofloat32": {"k": 0.0313, "b": 14.0207}
        }
    },
    "Maxs": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 20.0912},
            "float32tofloat32": {"k": 0.0141, "b": 20.0887}
        }
    },
    "Max": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0111, "b": 20.1200},
            "float32tofloat32": {"k": 0.0215, "b": 20.1333}
        }
    },
    "Mins": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 20.0896},
            "float32tofloat32": {"k": 0.0142, "b": 20.0876}
        }
    },
    "Min": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0111, "b": 20.1104},
            "float32tofloat32": {"k": 0.0215, "b": 20.1271}
        }
    },
    "Muls": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 23.1006},
            "float32tofloat32": {"k": 0.0142, "b": 23.0966},
            "int32toint32": {"k": 0.0180, "b": 20.7457}
        }
    },
    "Mul": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0110, "b": 23.1243},
            "float32tofloat32": {"k": 0.0206, "b": 23.2291}
        }
    },
    "Or": {
        "model_type": "SimpleLinear",
        "model_params": {
            "uint16touint16": {"k": 0.0132, "b": 12.4018}
        }
    },
    "PairReduceSum": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 37.159},
            "float32tofloat32": {"k": 0.1094, "b": 36.964}
        }
    },
    "Reciprocal": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0078, "b": 21.0076},
            "float32tofloat32": {"k": 0.0146, "b": 21.0639}
        }
    },
    "Relu": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0077, "b": 20.0173},
            "float32tofloat32": {"k": 0.0154, "b": 20.0189}
        }
    },
    "Rsqrt": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 21.0970},
            "float32tofloat32": {"k": 0.0143, "b": 21.0979}
        }
    },
    "Select": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0118, "b": 45.9656},
            "float32tofloat32": {"k": 0.0229, "b": 43.9906}
        }
    },
    "Sigmoid": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.1011, "b": 116.0436},
            "float32tofloat32": {"k": 0.1256, "b": 115.9747}
        }
    },
    "Sign": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0855, "b": 119.0821},
            "float32tofloat32": {"k": 0.1701, "b": 119.0656}
        }
    },
    "Sqrt": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0312, "b": 29.0056},
            "float32tofloat32": {"k": 0.0313, "b": 28.9961}
        }
    },
    "Sub": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0107, "b": 22.1226},
            "float32tofloat32": {"k": 0.0213, "b": 22.1254}
        }
    },
    "Tanh": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.1976, "b": 181.6919},
            "float32tofloat32": {"k": 0.1570, "b": 153.9298}
        }
    },
    "WholeReduceMax": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 21.0027},
            "float32tofloat32": {"k": 0.1094, "b": 20.0051}
        }
    },
    "WholeReduceMin": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 21.0056},
            "float32tofloat32": {"k": 0.1094, "b": 20.0068}
        }
    },
    "WholeReduceSum": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 35.0021},
            "float32tofloat32": {"k": 0.1094, "b": 32.0029}
        }
    },
    "LoadSmallBlk": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 2}
        }
    },
    "LoadUbStride": {
        "model_type": "LoadUbStride",
        "model_params": {
            "int64toint64": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 27.01, "a": 7.9052, "b": 7.3100, "hl": 0, "data_type_size": 2}
        }
    },
    "Load": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 27.01, "a": 9.9074, "b": 15.8960, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 27.01, "a": 9.9074, "b": 15.8960, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 27.01, "a": 9.9074, "b": 15.8960, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 27.01, "a": 9.9074, "b": 15.8960, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 27.01, "a": 9.9074, "b": 15.8960, "hl": 0, "data_type_size": 2}
        }
    },
    "LoadStride": {
        "model_type": "LoadStoreStrideFunc",
        "model_params": {
            "int64toint64": {"k": 0.07},
            "float16tofloat16": {"k": 0.07},
            "float32tofloat32": {"k": 0.07},
            "uint8touint8": {"k": 0.07},
            "bfloat16tobfloat16": {"k": 0.07}
        }
    },
    "Store": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 0, "data_type_size": 2}
        }
    },
    "StoreStride": {
        "model_type": "LoadStoreStrideFunc",
        "model_params": {
            "int64toint64": {"k": 0.02},
            "float16tofloat16": {"k": 0.02},
            "float32tofloat32": {"k": 0.02},
            "uint8touint8": {"k": 0.02},
            "bfloat16tobfloat16": {"k": 0.02}
        }
    },
    "StoreLargeBlk": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 1, "data_type_size": 8},
            "float16tofloat16": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 1, "data_type_size": 2},
            "float32tofloat32": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 1, "data_type_size": 4},
            "uint8touint8": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 1, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 12.09, "a": 9.96, "b": 3.79, "hl": 1, "data_type_size": 2}
        }
    },
    "StoreMiddleBlk": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 1.3, "a": 2, "b": 0, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 1.3, "a": 2, "b": 0, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 1.3, "a": 2, "b": 0, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 1.3, "a": 2, "b": 0, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 1.3, "a": 2, "b": 0, "hl": 0, "data_type_size": 2}
        }
    },
    "StoreSmallBlk": {
        "model_type": "StoreFunc",
        "model_params": {
            "int64toint64": {"ak": -0.101, "ab": -2.2, "bk": 8.89, "bb": 96.24, "h": 12.09, "data_type_size": 8},
            "float16tofloat16": {"ak": -0.101, "ab": -2.2, "bk": 8.89, "bb": 96.24, "h": 12.09, "data_type_size": 2},
            "float32tofloat32": {"ak": -0.101, "ab": -2.2, "bk": 8.89, "bb": 96.24, "h": 12.09, "data_type_size": 4},
            "uint8touint8": {"ak": -0.101, "ab": -2.2, "bk": 8.89, "bb": 96.24, "h": 12.09, "data_type_size": 1},
            "bfloat16tobfloat16": {"ak": -0.101, "ab": -2.2, "bk": 8.89, "bb": 96.24, "h": 12.09, "data_type_size": 2}
        }
    }
}
)";

const std::string *PerfParamTableV1::GetAscendCApiPerfTable() const {
  return &kParamV1Info;
}

PipeHeadPerfFunc PerfParamTableV1::GetPipeHeadPerfFunc(PipeType pipe_type) const {
  auto it = pipes_head_perf.find(pipe_type);
  if (it == pipes_head_perf.end()) {
    return &PerfParamTableV1::GetMTE2PipeHead;
  }
  return it->second;
}

PerfParamTableV1::PerfParamTableV1() {
  pipes_head_perf.emplace(PipeType::AIV_MTE2, &PerfParamTableV1::GetMTE2PipeHead);
  pipes_head_perf.emplace(PipeType::AIV_MTE3,
                          [](const std::vector<NodeInfo> &node_infos,
                             std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) -> Expr {
                            (void)node_infos;
                            (void)ternary_ops;
                            return kMte3PipeHeadcost;
                          });
  pipes_head_perf.emplace(PipeType::AIV_VEC,
                          [](const std::vector<NodeInfo> &node_infos,
                             std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) -> Expr {
                            (void)node_infos;
                            (void)ternary_ops;
                            return kVecheadcost;
                          });
}

Expr PerfParamTableV1::GetOpHeadCost() const {
  constexpr auto kOpHeadCost = 300.0;
  return CreateExpr(kOpHeadCost);
}
}  // namespace att

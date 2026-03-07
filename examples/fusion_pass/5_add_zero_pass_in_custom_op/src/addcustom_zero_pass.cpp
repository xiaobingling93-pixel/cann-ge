/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <math.h>
#include "es_all_ops.h"
#include "es_custom_ops.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

using namespace ge;
using namespace fusion;

/*
|o>--------------------------------------------------------------------------------------
                                                        0
                                                        |
           0                                        Identity
           |                                           |
|o>  a  Identity                                a  TensorMove           a     b
|o>    \ /              a      b                 \   /          ==>      \   /
|o> AddCustom  b   ==>   \    /                AddCustom   b           AddCustom
|o>      \    /         AddCustom                    \    /
|o>     AddCustom                                  AddCustom
|o>----------------------------------------------------------------------------------------
说明：考虑到不同版本torchair存在差异，为使该pass在不同版本下都能执行，本例定义了两个pattern,二者的替换结构相同。
*/

class AddCustomZeroPass : public PatternFusionPass {
 protected:
  std::vector<PatternUniqPtr> Patterns() override {
    std::cout << "Define pattern for AddCustomZeroPass" << std::endl;
    std::vector<PatternUniqPtr> patterns;

    auto graph_builder0 = es::EsGraphBuilder("pattern0");
    auto a0 = graph_builder0.CreateInput(0);
    auto b0 = graph_builder0.CreateInput(1);
    auto c0 = es::Const(graph_builder0);
    auto d0 = es::Identity(c0);
    auto add0 = es::AddCustom(a0, d0);
    auto add1 = es::AddCustom(add0, b0);
    auto graph0 = graph_builder0.BuildAndReset({add1});
    auto pattern0 = std::make_unique<Pattern>(std::move(*graph0));
    patterns.emplace_back(std::move(pattern0));

    auto graph_builder1 = es::EsGraphBuilder("pattern1");
    auto a1 = graph_builder1.CreateInput(0);
    auto b1 = graph_builder1.CreateInput(1);
    auto c1 = es::Const(graph_builder1);
    auto d1 = es::TensorMove(es::Identity(c1));
    auto add2 = es::AddCustom(a1, d1);
    auto add3 = es::AddCustom(add2, b1);
    auto graph1 = graph_builder1.BuildAndReset({add3});
    auto pattern1 = std::make_unique<Pattern>(std::move(*graph1));
    patterns.emplace_back(std::move(pattern1));

    return patterns;
  }

  // 判断符合pattern结构的拓扑中，Const是否为0
  bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define MeetRequirements for AddCustomZeroPass" << std::endl;
    std::vector<GNode> matched_nodes;
    matched_nodes = match_result->GetMatchedNodes();
    GNode matched_node;
    for (GNode node : matched_nodes) {
      AscendString type;
      node.GetType(type);
      if (type == "Const") {
        Tensor const_tensor;
        node.GetAttr("value", const_tensor);
        if (!IsTensorValueEqualToZero(const_tensor)) {
          return false;
        }
      }
    }
    return true;
  }

  GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define replacement for AddCustomZeroPass" << std::endl;
    auto replacement_graph_builder = es::EsGraphBuilder("replacement");
    auto r_a = replacement_graph_builder.CreateInput(0);
    auto r_b = replacement_graph_builder.CreateInput(1);

    auto add = es::AddCustom(r_a, r_b);

    return replacement_graph_builder.BuildAndReset({add});
  }

 private:
  bool IsTensorValueEqualToZero(const Tensor &tensor) {
    auto tensor_dtype = tensor.GetTensorDesc().GetDataType();
    switch (tensor_dtype) {
      case DT_FLOAT:
        if (std::fabs(*reinterpret_cast<const float *>(tensor.GetData())) < 1e-6) {
          return true;
        }
        return false;
      case DT_DOUBLE:
        if (std::fabs(*reinterpret_cast<const double *>(tensor.GetData())) < 1e-15) {
          return true;
        }
        return false;
      case DT_INT32:
        if (*reinterpret_cast<const int *>(tensor.GetData()) == 0) {
          return true;
        }
        return false;
      case DT_FLOAT16:
        if (std::fabs(*reinterpret_cast<const float *>(tensor.GetData())) < 1e-3) {
          return true;
        }
        return false;
      // 此处可以增加case支持更多数据类型
      default:
        std::cout << "Unsupported data type" << std::endl;
        return false;
    }
  }
};
REG_FUSION_PASS(AddCustomZeroPass).Stage(CustomPassStage::kBeforeInferShape);

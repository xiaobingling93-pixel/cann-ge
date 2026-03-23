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
#include <algorithm>
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/graph_rewriter.h"
#include "es_all_ops.h"

using namespace ge;
using namespace fusion;

bool IsNodeFedIntoRelu(const GNode &node) {
    if (node.GetOutputsSize() != 1) {return false;}
    auto [out_node, _] = node.GetOutDataNodesAndPortIndexs(0)[0];
    AscendString out_node_type;
    if (out_node->GetType(out_node_type) != GRAPH_SUCCESS) {
        std::cout << "GetType in IsNodeFedIntoRelu failed" << std::endl;
        return false;
    }
    if (out_node_type == "Relu") {
        return true;
    }
    std::cout << "find concat node not fed into relu" << std::endl;
    return false;
}

bool FindConcatNodesMeetRequirements(const GraphPtr &graph, std::vector<GNode> &concat_nodes) {
    for (auto &node: graph->GetDirectNode()) {
        AscendString node_type;
        if (node.GetType(node_type) != GRAPH_SUCCESS) {
            std::cout << "GetType in FindConcatNodesMeetRequirements failed" << std::endl;
            return false;
        }
        // 判断节点类型 && 该concat输出是否作为relu输入
        if (node_type == "ConcatV2" && IsNodeFedIntoRelu(node)) {
            concat_nodes.emplace_back(node);
        }
    }
    return true;
}

// 调用es接口构造替换结构replacement
GraphUniqPtr Replacement(const GNode &concat_node) {
    std::cout << "Define Replacement for MoveReluBeforeConcatPass" << std::endl;
    AscendString node_type;
    if (concat_node.GetType(node_type) != GRAPH_SUCCESS) {
        std::cout << "GetType in Replacement failed" << std::endl;
        return nullptr;
    }
    auto replacement_graph_builder = es::EsGraphBuilder("replacement");
    auto input_size = concat_node.GetInputsSize();
    if (node_type == "ConcatV2") {
        // 获取ConcatV2属性
        int N;
        if (concat_node.GetAttr("N", N) != GRAPH_SUCCESS) {
            std::cout << "GetAttr of concat node failed" << std::endl;
            return nullptr;
            }
        vector<es::EsTensorHolder> relued_tensors;
        for (size_t idx = 0; idx < input_size-1; ++idx) {
            auto input = replacement_graph_builder.CreateInput(idx);
            auto relu = es::Relu(input);
            relued_tensors.emplace_back(relu);
        }
        auto input_concat_dim = replacement_graph_builder.CreateInput(input_size-1);
        auto concat = es::ConcatV2(relued_tensors, input_concat_dim, N);
        auto replace_graph = replacement_graph_builder.BuildAndReset({concat});
        return replace_graph;
    }
    return nullptr;
}

// 构造需要被替换的子图边界
std::unique_ptr<SubgraphBoundary> ConstructSubgraphBoundary(const GNode &concat_node) {
    auto input_size = concat_node.GetInputsSize();
    std::unique_ptr<SubgraphBoundary> boundary = std::make_unique<SubgraphBoundary>();
    for (size_t input_idx = 0; input_idx < input_size; ++input_idx) {
        SubgraphInput subgraph_input;
        subgraph_input.AddInput({concat_node, int64_t(input_idx)});
        if (boundary->AddInput(input_idx, std::move(subgraph_input)) != SUCCESS) { return nullptr; }
    }
    // 该场景下concat节点输出数量是1，且单输入单引用
    auto [output_node, _] = concat_node.GetOutDataNodesAndPortIndexs(0)[0];
    AscendString out_node_type;
    if (output_node->GetType(out_node_type) != GRAPH_SUCCESS) {
        std::cout << "GetType in ConstructSubgraphBoundary failed" << std::endl;
        return nullptr;
    }
    if (out_node_type != "Relu") {
        std::cout << "Output node of concat is not target type" << std::endl;
        std::cout << out_node_type.GetString() << std::endl;
        return nullptr;
    }
    // 该场景下Relu节点输出数量是1，且单输出单引用
    SubgraphOutput subgraph_output({*output_node, 0});
    if (boundary->AddOutput(0, std::move(subgraph_output)) != SUCCESS) { return nullptr; }
    return boundary;
}

bool MoveReluBeforeConcat(const GNode &concat_node) {
    // 根据concat节点构造替换结构replacement
    const auto replacement = Replacement(concat_node);
    if (!replacement) {
        std::cout << "Define replacement failed" << std::endl;
        return false;
    }
    // 构造被替换的子图边界boundary
    auto boundary = ConstructSubgraphBoundary(concat_node);
    if (boundary == nullptr) {
        std::cout << "ConstructSubgraphBoundary failed, boundary is nullptr" << std::endl;
        return false;
    }
    // 替换
    if (SubgraphRewriter::Replace(*boundary, *replacement) != SUCCESS) {
        std::cout << "Replace failed" << std::endl;
        return false;
    }
    std::cout << "Replacement of MoveReluBeforeConcatPass succeeded" << std::endl;
    return true;
}

/*
|o>-----------------------------------
|o>    \  |  /         \     |     /
|o>     Concat         ReLu  ReLu ReLu
|o>       |       ==>    \   |   /
|o>      ReLu             Concat
|o>-----------------------------------
说明：本例识别上图中左边的Concat&ReLu结构并通过图修改接口将ReLu移至Concat前
实现逻辑：
1.通过Graph接口遍历当前图节点搜索符合条件的Concat。
2.定义替换结构replacement。
3.构造需要被替换的图边界boundary。
4.调用SubgraphRewriter::Replace进行改图。
*/
class MoveReluBeforeConcatPass : public FusionBasePass {
public:
    Status Run(GraphPtr &graph, CustomPassContext &pass_context) override {
        std::cout << "MoveReluBeforeConcatPass" << std::endl;
        // 备份原图用于回退
        Graph origin_graph = *graph;
        // 遍历节点获取符合条件的concat节点
        std::vector<GNode> concat_nodes;
        if (!FindConcatNodesMeetRequirements(graph, concat_nodes)) {
            std::cout << "FindConcatNodesMeetRequirements failed" << std::endl;
            *graph = origin_graph;
            return FAILED;
        }
        if (concat_nodes.empty()) {
            std::cout << "No concat nodes found" << std::endl;
            return SUCCESS;
        }
        // 遍历每个符合条件的concat_node，移动relu
        for (auto &node: concat_nodes) {
            if (!MoveReluBeforeConcat(node)) {
                std::cout << "MoveReluBeforeConcat failed" << std::endl;
                *graph = origin_graph;
                return FAILED;
            }
        }
        return SUCCESS;
    }
};

REG_FUSION_PASS(MoveReluBeforeConcatPass).Stage(CustomPassStage::kBeforeInferShape);
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
#include <queue>
#include <algorithm>
#include "ge/fusion/pass/pattern_fusion_pass.h"

using namespace ge;
using namespace fusion;

// 需要匹配的卷积节点类型
const std::vector<AscendString> TargetTypes{"Conv2D", "Conv2DV2"};
constexpr int64_t dataIdxOfTransposeInputs = 0;
constexpr int64_t permIdxOfTransposeInputs = 1;
constexpr int64_t idxOfTransposeOutput = 0;

template<typename T>
bool JudgeTransposeConstData(uint8_t *const_data, size_t data_len, int &cnt) {
    if (const_data == nullptr || data_len == 0 || (data_len % sizeof(T)) != 0) {
        std::cout << "GetConstData error" << std::endl;
        return false;
    }
    std::vector<T> result;
    T *T_buffer = reinterpret_cast<T*>(const_data);
    result.assign(T_buffer, T_buffer + (data_len / sizeof(T)));
    if ((cnt == 0 && result == std::vector<T>{0,2,3,1}) ||
        (cnt == 1 && result == std::vector<T>{0,3,1,2})) {
        ++cnt;
        return true;
        }
    return false;
}

bool IsMatchAnyOfType(const AscendString &op_type) {
    return std::any_of(
        TargetTypes.cbegin(), TargetTypes.cend(),
        [&op_type](const AscendString &target_type) {return op_type == target_type;});
}

bool FindNCHWConvNodes(const GraphPtr &graph, std::vector<GNode> &conv_nodes) {
    for (auto &node: graph->GetDirectNode()) {
        AscendString node_type, node_format;
        if (node.GetType(node_type) != GRAPH_SUCCESS) {return false;}
        if (IsMatchAnyOfType(node_type)
            && node.GetAttr("data_format", node_format) == GRAPH_SUCCESS && node_format == "NCHW") { // 判断node类型 && 获取format属性 && 判断是否NCHW
            conv_nodes.emplace_back(node);
        }
    }
    return !conv_nodes.empty();
}

bool JudgeTransposeNode(const GNodePtr &node_ptr, int &cnt) {
    Tensor const_tensor;
    if (node_ptr->GetInputConstData(permIdxOfTransposeInputs, const_tensor) != GRAPH_SUCCESS) {
        return false;
    }
    uint8_t *const_data = const_tensor.GetData();
    size_t data_len = const_tensor.GetSize();
    auto tensor_desc = const_tensor.GetTensorDesc();
    auto t_datatype = tensor_desc.GetDataType();
    if (t_datatype == DataType::DT_INT32) {
        return JudgeTransposeConstData<int32_t>(const_data, data_len, cnt);
    }
    if (t_datatype == DataType::DT_INT64) {
        return JudgeTransposeConstData<int64_t>(const_data, data_len, cnt);
    }
    return false;
}

bool RemoveTransposeAndRelink(const GraphPtr &graph, const GNodePtr &node_ptr) {
    auto [data_node, data_output_index] = node_ptr->GetInDataNodesAndPortIndexs(dataIdxOfTransposeInputs);
    auto [perm_node, perm_output_index] = node_ptr->GetInDataNodesAndPortIndexs(permIdxOfTransposeInputs);
    if (graph->RemoveEdge(*data_node, data_output_index, *node_ptr, dataIdxOfTransposeInputs) != GRAPH_SUCCESS ||
        graph->RemoveEdge(*perm_node, perm_output_index, *node_ptr, permIdxOfTransposeInputs) != GRAPH_SUCCESS ) {
        std::cout << "Remove input edges failed" << std::endl;
        return false;
    }
    for (auto &[out_node, out_input_index] : node_ptr->GetOutDataNodesAndPortIndexs(idxOfTransposeOutput)) {
        if (out_node != nullptr) {
            if (graph->RemoveEdge(*node_ptr, idxOfTransposeOutput, *out_node, out_input_index) == GRAPH_SUCCESS &&
                graph->AddDataEdge(*data_node, data_output_index, *out_node, out_input_index) == GRAPH_SUCCESS) {
                std::cout << "Remove output edges success" << std::endl;
                continue;
            }
            return false;
        }
    }
    if (graph->RemoveNode(*perm_node) != GRAPH_SUCCESS || graph->RemoveNode(*node_ptr) != GRAPH_SUCCESS) {
        return false;
    }
    return true;
}

bool DeleteTransposePairBehindIfExist(const GraphPtr &graph, const GNode &conv_node) {
    // 广度遍历寻找conv_node后的transpose
    std::queue<GNodePtr> bfs_node_queue;
    int transpose_cnt = 0;
    auto conv_output_size = conv_node.GetOutputsSize();

    for (size_t idx = 0; idx < conv_output_size; ++idx) {
        std::vector<std::pair<GNodePtr, int32_t>> output_nodes_idxes = conv_node.GetOutDataNodesAndPortIndexs(idx);
        for (auto pair : output_nodes_idxes) {
            bfs_node_queue.push(pair.first);
        }
    }
    while (!bfs_node_queue.empty()) {
        auto node_ptr = bfs_node_queue.front();
        bfs_node_queue.pop();
        // 该节点的所有输出节点入队
        auto output_size = node_ptr->GetOutputsSize();
        for (size_t idx = 0; idx < output_size; ++idx) {
            std::vector<std::pair<GNodePtr, int32_t>> output_nodes_idxes = node_ptr->GetOutDataNodesAndPortIndexs(idx);
            for (auto pair : output_nodes_idxes) {
                bfs_node_queue.push(pair.first);
            }
        }
        // 判断节点是否是transpose && 判断perm
        // 如果 cnt == 0 && perm == {0,2,3,1}，成功找到第1个，++ cnt
        // 如果 cnt == 1 && perm == {0,3,1,2}，成功找到第2个，++ cnt
        AscendString node_type;
        if (node_ptr->GetType(node_type) != GRAPH_SUCCESS) {return false;}
        if (node_type == "Transpose" && JudgeTransposeNode(node_ptr,transpose_cnt)) {
            // 删除节点(包括const)并重新连边
            if (!RemoveTransposeAndRelink(graph, node_ptr)) {
                std::cout << "RemoveTransposeAndRelink failed "<< std::endl;
                return false;
            };
            // 如果cnt == 2，直接返回true
            if (transpose_cnt == 2) {return true;}
        }
    }
    return true;
}

/*
|o>-----------------------------------
|o>          |
|o>      Conv[NCHW]            |
|o>          |       ==>    Conv[NCHW]
|o>      Transpose              |
|o>         ...                ...
|o>      Transpose
|o>-----------------------------------
说明：本例假设的业务场景为一个输入输出data format为NCHW的网络中，
主要的计算使用NHWC，于是在经过上图的Conv[NCHW]后需要Transpose到NHWC后
完成计算,并再次Transpose为NCHW。现希望将网络通过pass改造成输入输出为NHWC。
为了达到以上目的，识别上图中左边data_format为NCHW的Conv节点，修改为NHWC并尝试删除
该Conv后的一对Transpose("尝试删除"的原因是假设场景中有多个Conv[NCHW]共用一对Transpose)。
实现逻辑：
1.通过Graph接口遍历当前图节点搜索符合条件的Conv。
2.使用SetAttr修改被选中Conv的data_format属性。
3.广度遍历搜索Conv后的Transpose。
4.使用RemoveEdge、AddDataEdge等删除符合条件的Transpose。
*/
class ConvTransFormatPass : public FusionBasePass {
public:
    Status Run(GraphPtr &graph, CustomPassContext &pass_context) override {
        // 备份原图用于回退
        std::cout << "ConvTransFormatPass is starting" << std::endl;
        Graph origin_graph = *graph;
        std::vector<GNode> conv_nodes;
        if (!FindNCHWConvNodes(graph, conv_nodes)) {
            std::cout << "Graph has no Conv node in NCHW format" << std::endl;
            return SUCCESS;
        }
        for (auto &node: conv_nodes) {
            AscendString format_NHWC = "NHWC";
            if (node.SetAttr("data_format", format_NHWC) != GRAPH_SUCCESS) {
                std::cout << "Modify format of node failed" << std::endl;
                *graph = origin_graph;
                return FAILED;
            }
            if (!DeleteTransposePairBehindIfExist(graph, node)) {
                std::cout << "DeleteTransposePairBehindIfExist failed" << std::endl;
                *graph = origin_graph;
                return FAILED;
            }
        }
        std::cout << "ConvTransFormatPass completed" << std::endl;
        return SUCCESS;
    }
};

REG_FUSION_PASS(ConvTransFormatPass).Stage(CustomPassStage::kBeforeInferShape);
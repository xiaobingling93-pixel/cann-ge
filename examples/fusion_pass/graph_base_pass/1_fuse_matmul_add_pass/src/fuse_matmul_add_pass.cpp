/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "register_custom_pass.h"
#include "all_ops.h"

using namespace std;
using namespace ge;

namespace {
constexpr const char *kOpNameAdd = "add";
constexpr const char *kOpNameMatMul = "matmul";
constexpr const char *kOpNameGEMM = "gemm";
constexpr const char *kOpNameAlpha = "alpha";
constexpr const char *kOpNameBeta = "beta";
constexpr const char *kAttrNameTransposeA = "transpose_a";
constexpr const char *kAttrNameTransposeB = "transpose_b";
constexpr int32_t kIndex0 = 0;
constexpr int32_t kIndex1 = 1;
constexpr int32_t kIndex2 = 2;
constexpr int32_t kIndex3 = 3;
constexpr int32_t kIndex4 = 4;

bool FindNodes(GraphPtr &graph, GNode &src_node, GNode &dst_node) {
    auto all_nodes = graph->GetAllNodes();
    bool find_src_node = false;
    bool find_dst_node = false;
    for (auto &node: all_nodes) {
        AscendString node_name;
        auto ret = node.GetName(node_name);
        if (node_name == kOpNameMatMul) {
            src_node = node;
            find_src_node = true;
            cout << "Find src node: MatMul." << endl;
        } else if (node_name == kOpNameAdd) {
            dst_node = node;
            find_dst_node = true;
            cout << "Find dst node: Add." << endl;
        }
    }
    return (find_src_node && find_dst_node);
}

bool CheckNodesHaveEdge(GraphPtr &graph, const GNode &src_node, const GNode &dst_node) {
    for (auto &[out_node, _]: src_node.GetOutDataNodesAndPortIndexs(kIndex0)) {
        AscendString node_name;
        auto ret = out_node->GetName(node_name);
        if (node_name == kOpNameAdd) {
            return true;
        }
    }
    return false;
}

void CreateGEMMNode(GraphPtr &graph, const GNode &src_node, GNode &node_gemm) {
    bool transpose_a = false;
    bool transpose_b = false;
    src_node.GetAttr(kAttrNameTransposeA, transpose_a);
    src_node.GetAttr(kAttrNameTransposeB, transpose_b);
    constexpr float kValue1 = 1;
    TensorDesc alpha_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
    Tensor alpha_tensor(alpha_desc, reinterpret_cast<const uint8_t *>(&kValue1), sizeof(float));
    auto alpha = op::Const(kOpNameAlpha).set_attr_value(alpha_tensor);
    TensorDesc beta_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
    Tensor beta_tensor(beta_desc, reinterpret_cast<const uint8_t *>(&kValue1), sizeof(float));
    auto beta = op::Const(kOpNameBeta).set_attr_value(beta_tensor);

    auto gemm = op::GEMM(kOpNameGEMM);
    gemm.set_attr_transpose_a(transpose_a)
        .set_attr_transpose_b(transpose_b);
    gemm.update_input_desc_alpha(alpha_desc);
    gemm.update_input_desc_beta(beta_desc);

    auto node_alpha = graph->AddNodeByOp(alpha);
    auto node_beta = graph->AddNodeByOp(beta);
    node_gemm = graph->AddNodeByOp(gemm);

    auto ret = graph->AddDataEdge(node_alpha, kIndex0, node_gemm, kIndex3);
    ret = graph->AddDataEdge(node_beta, kIndex0, node_gemm, kIndex4);
}

bool AddInputsAndOutputs(GraphPtr &graph, const GNode &src_node, const GNode &dst_node, GNode &node_gemm) {
    auto [a, a_output_index] = src_node.GetInDataNodesAndPortIndexs(kIndex0);
    auto [b, b_output_index] = src_node.GetInDataNodesAndPortIndexs(kIndex1);
    int32_t add_node_c_input_index = -1;
    for (size_t i = 0; i < dst_node.GetInputsSize(); ++i) {
        auto [in_node, _] = dst_node.GetInDataNodesAndPortIndexs(i);
        AscendString node_name;
        auto ret = in_node->GetName(node_name);
        if (node_name != kOpNameMatMul) {
            add_node_c_input_index = i;
            break;
        }
    }
    if (add_node_c_input_index == -1) {
        return false;
    }
    auto [c, c_output_index] = dst_node.GetInDataNodesAndPortIndexs(add_node_c_input_index);
    auto ret = graph->AddDataEdge(*a, a_output_index, node_gemm, kIndex0);
    if (ret != GRAPH_SUCCESS) {
        return false;
    }
    ret = graph->AddDataEdge(*b, b_output_index, node_gemm, kIndex1);
    ret = graph->AddDataEdge(*c, c_output_index, node_gemm, kIndex2);

    TensorDesc input_desc_a;
    ret = src_node.GetInputDesc(kIndex0, input_desc_a);
    ret = node_gemm.UpdateInputDesc(kIndex0, input_desc_a);

    TensorDesc input_desc_b;
    ret = src_node.GetInputDesc(kIndex1, input_desc_b);
    ret = node_gemm.UpdateInputDesc(kIndex1, input_desc_b);

    TensorDesc input_desc_c;
    ret = dst_node.GetInputDesc(add_node_c_input_index, input_desc_c);
    ret = node_gemm.UpdateInputDesc(kIndex2, input_desc_c);

    TensorDesc output_desc_y;
    ret = dst_node.GetOutputDesc(kIndex0, output_desc_y);
    ret = node_gemm.UpdateOutputDesc(kIndex0, output_desc_y);
    return true;
}

void RemoveOldNodesEdgesAndAddGemmOutput(GraphPtr &graph, GNode &src_node, GNode &dst_node, GNode &node_gemm) {
    vector<GNode> node_vec{src_node, dst_node};
    for (auto &node: node_vec) {
        for (size_t i = 0; i < node.GetInputsSize(); ++i) {
            auto [in_node, in_id] = node.GetInDataNodesAndPortIndexs(i);
            if (in_node != nullptr) {
                auto ret = graph->RemoveEdge(*in_node, in_id, node, i);
            }
        }
    }

    for (auto &[out_node, out_id]: dst_node.GetOutDataNodesAndPortIndexs(kIndex0)) {
        if (out_node != nullptr) {
            auto ret = graph->RemoveEdge(dst_node, kIndex0, *out_node, out_id);
            ret = graph->AddDataEdge(node_gemm, kIndex0, *out_node, out_id);
        }
    }

    for (auto &node: node_vec) {
        auto ret = graph->RemoveNode(node);
    }
}
} // namespace

// |o>-----------------------------------
// |o>    a  b
// |o>    \ /              a   b    c
// |o>   MatMul  c   ==>   \   |   /
// |o>     \    /            GEMM
// |o>      Add
// |o>-----------------------------------
// 融合说明：本例识别上图中左边的MatMul+Add结构并通过图修改接口替换为右边的单个GEMM节点
// 改图接口返回值说明：本文件中的改图接口需要判断返回值, 基于可读性考虑除了pass入口函数外其他函数中的改图接口只接收返回值
// 但不增加返回值处理代码。如需判断返回值，可配合使用custom_context.SetErrorMessage("xxx")方法
graphStatus FuseMatMulAndAddPass(GraphPtr &graph, CustomPassContext &custom_context) {
    cout << "FuseMatMulAndAddPass begin." << endl;
    GNode src_node;
    GNode dst_node;
    // 1.遍历所有节点，寻找MatMul和Add节点
    if (!FindNodes(graph, src_node, dst_node)) {
        cout << "Do not find MatMul or Add node." << endl;
        return GRAPH_SUCCESS;
    }

    // 2.判断MatMul和Add节点是否有连边关系
    if (!CheckNodesHaveEdge(graph, src_node, dst_node)) {
        cout << "There is no edge between src and dst node." << endl;
        return GRAPH_SUCCESS;
    }

    // 3.创建和添加GEMM节点
    GNode node_gemm;
    CreateGEMMNode(graph, src_node, node_gemm);

    // 4.添加新节点的输入输出
    if (!AddInputsAndOutputs(graph, src_node, dst_node, node_gemm)) {
        custom_context.SetErrorMessage("Add inputs and outputs failed.");
        return -1;
    }

    // 5.删除旧节点和其连边关系，连接新GEMM节点和输出节点
    RemoveOldNodesEdgesAndAddGemmOutput(graph, src_node, dst_node, node_gemm);

    cout << "FuseMatMulAndAddPass end." << endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("FuseMatMulAndAddPass").CustomPassFn(FuseMatMulAndAddPass);
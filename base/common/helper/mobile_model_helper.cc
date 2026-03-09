/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/helper/mobile_model_helper.h"

#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <fstream>
#include <regex>
#include <set>
#include <climits>

#include "common/checker.h"
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "common/op_so_store/op_so_store_utils.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "framework/omg/version.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/unfold/graph_unfolder.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/omg/omg_inner_types.h"
#include "mmpa/mmpa_api.h"
#include "common/proto_util/proto_util.h"
#include "graph/ge_local_context.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/manager/graph_var_manager.h"
#include "common/math/math_util.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/helper/file_saver.h"
#include "common/model/model_introduction.h"
#include "common/model/model_compress_manager.h"
#include "common/host_resource_center/host_resource_serializer.h"
#include "graph/utils/math_util.h"
#include "ge_context.h"
#include "proto/task.pb.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "base/err_msg.h"
#include "graph_metadef/graph/debug/ge_util.h"

#include "mobile/base_buffer.h"
#include "mobile/compiled_model.h"
#include "mobile/model_file_saver.h"
#include "mobile/compiled_target.h"
#include "mobile/compatible_info_v2.h"
#include "mobile/kernelbin_info.h"
#include "proto/ge_ir_mobile.pb.h"
#include "proto/task_mobile.pb.h"


namespace {

constexpr uint64_t PER_U64_BYTES = 8;
constexpr int32_t RT_MODEL_TASK_KERNEL = 0;
constexpr int32_t ASCENDC_AI_CORE = 6;
constexpr int32_t PER_ARG_BYTES = 8;
constexpr uint32_t MOBILE_RT_DEV_BINARY_MAGIC_ELF = 0x43554245U;

class ShapeInfo {
public:
    int64_t dim_num;
    std::vector<int64_t> dims;
    int64_t offset;
    std::string ToString() const
    {
        std::string msg = "";
        msg += "dim_num: " + std::to_string(dim_num) + ", shape: ";
        for (const auto& d: dims) {
            msg += std::to_string(d) + " ";
        }
        msg += ", offset: " + std::to_string(offset);
        return msg;
    }
};

class DynamicInputsOutputsShapeInfo {
public:
    std::vector<std::vector<ShapeInfo>>& dynamic_inputs_shape;
    std::vector<std::vector<ShapeInfo>>& dynamic_outputs_shape;

    const std::vector<ShapeInfo>& GetShapeInfo(const std::string& io_type, uint64_t group_index) const
    {
        static std::vector<ShapeInfo> dynamic_shape_empty;
        if ((io_type == "i" && static_cast<size_t>(group_index) >= dynamic_inputs_shape.size()) ||
            (io_type == "o" && static_cast<size_t>(group_index) >= dynamic_outputs_shape.size())) {
            GELOGE(ge::FAILED, "[Mobile] group index: %d , io_type: %s, in size: %d, out size: %d, is invaild, return.",
                group_index, io_type.c_str(), dynamic_inputs_shape.size(), dynamic_outputs_shape.size());
            return dynamic_shape_empty;
        }
        if (io_type == "i") {
            return dynamic_inputs_shape[group_index];
        } else if (io_type == "o") {
            return dynamic_outputs_shape[group_index];
        } else {
            GELOGE(ge::FAILED, "[Mobile] io_type is not support, failed.");
        }
        return dynamic_shape_empty;
    }
};

uint64_t ConstructSecondaryAddrContext(std::vector<uint8_t>& flowtable, const std::vector<ShapeInfo>& shape_infos)
{
    // cal real size
    // ptr_offset
    uint64_t real_size = PER_U64_BYTES;
    // shape info(dim + shape)
    for (const auto& shape_info: shape_infos) {
        GE_ASSERT_TRUE((static_cast<uint64_t>(shape_info.dim_num) <= UINT64_MAX / PER_U64_BYTES) &&
            (PER_U64_BYTES <= UINT64_MAX - (static_cast<uint64_t>(shape_info.dim_num) * PER_U64_BYTES)) &&
            (real_size <= UINT64_MAX - (PER_U64_BYTES + (static_cast<uint64_t>(shape_info.dim_num) * PER_U64_BYTES))),
            "[Mobile] overflow, failed.");
        real_size += PER_U64_BYTES + static_cast<uint64_t>(shape_info.dim_num) * PER_U64_BYTES;
    }
    // ptr
    real_size += static_cast<uint64_t>(shape_infos.size()) * PER_U64_BYTES;
    GELOGI("[Mobile] real size: %d", real_size);

    std::vector<uint64_t> tlv(real_size / PER_U64_BYTES);
    uint64_t tlv_index = 0;
    // ptr_offset
    tlv[tlv_index] = PER_U64_BYTES;
    for (const auto& shape_info: shape_infos) {
        if (static_cast<uint64_t>(shape_info.dim_num) > (UINT64_MAX / PER_U64_BYTES) ||
            (static_cast<uint64_t>(shape_info.dim_num) * PER_U64_BYTES) > (UINT64_MAX - PER_U64_BYTES)) {
            GELOGE(ge::FAILED, "[Mobile] cal overflow, failed.");
            return 0;
        }
        tlv[tlv_index] += PER_U64_BYTES + static_cast<uint64_t>(shape_info.dim_num) * PER_U64_BYTES;
    }
    GELOGI("[Mobile] tlv ptr offset: 0x%llx", tlv[tlv_index]);
    tlv_index++;
    GELOGI("[Mobile] tlv_index after fill ptr offset: %d", tlv_index);
    // shape info(dim + shape)
    for (size_t i = 0; i < shape_infos.size(); i++) {
        constexpr uint64_t shift_32_bits = 32;
        // dim
        tlv[tlv_index++] = static_cast<uint64_t>(i) << shift_32_bits |  static_cast<uint64_t>(shape_infos[i].dim_num);
        // shape
        for (const auto& s: shape_infos[i].dims) {
            tlv[tlv_index++] = static_cast<uint64_t>(s);
        }
        GELOGI("[Mobile] tlv_index after shape index: %d , tlv_index: %d", i, tlv_index);
    }
    // ptr
    for (size_t i = 0; i < shape_infos.size(); i++) {
        // add 2G
        tlv[tlv_index++] = static_cast<uint64_t>(0x80000000U) | static_cast<uint64_t>(shape_infos[i].offset);
        GELOGI("[Mobile] tlv_index after ptr index: %d , tlv_index: %d", i, tlv_index);
    }
    const uint8_t* tlv_data = reinterpret_cast<uint8_t*>(tlv.data());
    (void)flowtable.insert(flowtable.end(), tlv_data, &tlv_data[real_size]);
    return real_size;
}

ge::Status GetDynamicInputsInfo(const ge::NodePtr& node,
    std::vector<int64_t>& input_offset, std::vector<std::vector<ShapeInfo>>& dynamic_inputs_shape)
{
    std::vector<std::vector<int64_t>> dynamic_inputs_indexes;
    const std::string dynamic_inputs_indexes_str = "_dynamic_inputs_indexes";
    (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), dynamic_inputs_indexes_str, dynamic_inputs_indexes);
    input_offset = node->GetOpDesc()->GetInputOffset();
    for (const auto& i_offset: input_offset) {
        GELOGI("[Mobile] input offset: %d ", i_offset);
    }
    GELOGI("[Mobile] dynamic inputs group count: %d ", dynamic_inputs_indexes.size());
    for (size_t i = 0; i < dynamic_inputs_indexes.size(); i++) {
        GELOGI("[Mobile] dynamic inputs group: %d", i);
        for (const auto& d: dynamic_inputs_indexes[i]) {
            GELOGI("[Mobile] input indexes: %d ", d);
        }
        std::vector<ShapeInfo> shape_infos;
        for (const auto& d: dynamic_inputs_indexes[i]) {
            if (static_cast<size_t>(d) >= node->GetOpDesc()->GetInputsSize() ||
                static_cast<size_t>(d) >= input_offset.size()) {
                GELOGE(ge::FAILED, "[Mobile] dynamic inputs index: %d  is invalid, return", d);
                return ge::FAILED;
            }
            ShapeInfo shape_info;
            shape_info.dim_num = static_cast<int64_t>(node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(d)).GetShape().GetDimNum());
            shape_info.dims = node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(d)).GetShape().GetDims();
            shape_info.offset = input_offset[static_cast<uint64_t>(d)];
            shape_infos.push_back(shape_info);
        }
        dynamic_inputs_shape.push_back(shape_infos);
    }
    for (size_t i = 0; i < dynamic_inputs_shape.size(); i++) {
        GELOGI("[Mobile] dynamic inputs group: %d", i);
        for (const auto& s: dynamic_inputs_shape[i]) {
            GELOGI("[Mobile] %s", s.ToString().c_str());
        }
    }
    return ge::SUCCESS;
}

ge::Status GetDynamicOutputsInfo(const ge::NodePtr& node,
    std::vector<int64_t>& output_offset, std::vector<std::vector<ShapeInfo>>& dynamic_outputs_shape)
{
    std::vector<std::vector<int64_t>> dynamic_outputs_indexes;
    const std::string dynamic_outputs_indexes_str = "_dynamic_outputs_indexes";
    (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), dynamic_outputs_indexes_str, dynamic_outputs_indexes);
    output_offset = node->GetOpDesc()->GetOutputOffset();
    for (const auto& o_offset: output_offset) {
        GELOGI("[Mobile] output offset: %d ", o_offset);
    }
    GELOGI("[Mobile] dynamic outputs group count: %d", dynamic_outputs_indexes.size());
    for (size_t i = 0; i < dynamic_outputs_indexes.size(); i++) {
        GELOGI("[Mobile] dynamic outputs group: %d", i);
        for (const auto& d: dynamic_outputs_indexes[i]) {
            GELOGI("[Mobile] output indexes: %d ", d);
        }
        std::vector<ShapeInfo> shape_infos;
        for (const auto& d: dynamic_outputs_indexes[i]) {
            if (static_cast<size_t>(d) >= node->GetOpDesc()->GetOutputsSize() ||
                static_cast<size_t>(d) >= output_offset.size()) {
                GELOGE(ge::FAILED, "[Mobile] dynamic outputs index: %d  is invalid, return", d);
                return ge::FAILED;
            }
            ShapeInfo shape_info;
            shape_info.dim_num = static_cast<int64_t>(node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(d)).GetShape().GetDimNum());
            shape_info.dims = node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(d)).GetShape().GetDims();
            shape_info.offset = output_offset[static_cast<uint64_t>(d)];
            shape_infos.push_back(shape_info);
        }
        dynamic_outputs_shape.push_back(shape_infos);
    }
    for (size_t i = 0; i < dynamic_outputs_shape.size(); i++) {
        GELOGI("[Mobile] dynamic outputs group: %d", i);
        for (const auto& s: dynamic_outputs_shape[i]) {
            GELOGI("[Mobile] %s", s.ToString().c_str());
        }
    }
    return ge::SUCCESS;
}

class KernelArgInfoWithShapeInfo {
public:
    // 'i' is input
    // 'o' is output
    // 'ws' is workspace
    std::string type;
    // '*' is x1 x2 y1 y2 ...
    // ''  is [x1,x2,x3], [y1,y2,y3], ...
    // 'desc*' is x1 shape, x2 shape, x1 ptr, x2, ptr, y1 shape, y2 shape, y1 ptr, y2, ptr, ...
    // 'desc' is [x1 shape, x2 shape, x1 ptr, x2, ptr], [y1 shape, y2 shape, y1 ptr, y2, ptr], ...
    std::string format;
    int32_t group_index;
    std::set<int32_t> arg_indexes;
    int32_t add;

    KernelArgInfoWithShapeInfo(const std::string& type_in, const std::string& format_in, int32_t group_index_in, int32_t add_in)
        : type(type_in), format(format_in), group_index(group_index_in), arg_indexes({}), add(add_in) {}

    std::string ToString() const
    {
        std::string msg = "type: " + type +
            ", format: " + format +
            ", add: " + std::to_string(add) +
            ", group_index: " + std::to_string(group_index) +
            ", arg_indexes: ";
        for (const auto& i: arg_indexes) {
            msg += std::to_string(i) + " ";
        }
        return msg;
    }
};

ge::Status ParseArgsFormat(const std::string& args_format,
    std::vector<KernelArgInfoWithShapeInfo>& arg_infos_with_shape_info)
{
    std::regex pattern("\\{[a-z0-9_\\*]+\\}");
    std::smatch sm;
    auto s_start = args_format.cbegin();
    int32_t input_group_index = 0;
    int32_t output_group_index = 0;
    int32_t add = 0;
    while (std::regex_search(s_start, args_format.cend(), sm, pattern)) {
        if (sm.size() != 1) {
            GELOGW("[Mobile] sm size is not 1, break.");
            break;
        }
        std::string arg_format = std::string(sm[0]);
        if (arg_format.empty()) {
            GELOGW("[Mobile] arg format is empty, break.");
            break;
        }
        GELOGI("[Mobile] arg_format string is: %s", arg_format.c_str());
        if (arg_format.find("desc") != std::string::npos) {
            std::regex pattern_with_shape_info("([a-z])_desc([0-9]+)(\\*?)");
            std::smatch sm_with_shape_info;
            (void)std::regex_search(arg_format, sm_with_shape_info, pattern_with_shape_info);
            constexpr size_t shape_info_size_needed = 4;
            GE_ASSERT_TRUE(sm_with_shape_info.size() == shape_info_size_needed, "[Mobile] shape info size need > 4.");
            for (size_t idx = 0; idx < shape_info_size_needed; idx++) {
                GELOGI("[Mobile] -> [%d]: %s", idx, sm_with_shape_info[idx].str().c_str());
            }
            GE_ASSERT_TRUE(sm_with_shape_info[shape_info_size_needed - 1UL] != "*", "[Mobile] not support desc*.");
            int32_t group_index = -1;
            if (sm_with_shape_info[1] == "i") {
                group_index = input_group_index++;
            } else if (sm_with_shape_info[1] == "o") {
                group_index = output_group_index++;
            } else {
                GELOGE(ge::FAILED, "[Mobile] sm_with_shape_info[1] is not support, failed.");
                return ge::FAILED;
            }
            arg_infos_with_shape_info.push_back(KernelArgInfoWithShapeInfo(sm_with_shape_info[1], "desc", group_index, add));
        } else {
            add++;
            GELOGI("[Mobile] arg without shape info, skip");
        }
        s_start = sm.suffix().first;
    }
    for (const auto& i: arg_infos_with_shape_info) {
        GELOGI("[Mobile] %s", i.ToString().c_str());
    }
    return ge::SUCCESS;
}

ge::Status SetArgInfosArgsIndex(std::vector<KernelArgInfoWithShapeInfo>& arg_infos_with_shape_info,
    DynamicInputsOutputsShapeInfo& dynamic_inputsoutputs_shapeinfo)
{
    auto& dynamic_inputs_shape = dynamic_inputsoutputs_shapeinfo.dynamic_inputs_shape;
    auto& dynamic_outputs_shape = dynamic_inputsoutputs_shapeinfo.dynamic_outputs_shape;

    int32_t args_index = 0;
    for (auto& arg_info: arg_infos_with_shape_info) {
        int32_t group_index = arg_info.group_index;
        std::string io_type = arg_info.type;
        std::vector<ShapeInfo>* dynamic_inout_shape = nullptr;
        if (io_type == "i") {
            GE_ASSERT_TRUE(static_cast<size_t>(group_index) < dynamic_inputs_shape.size(),
                "[Mobile] group_index is invaild, group_index: %d, dynamic_inputs_shape.size: %d",
                group_index, dynamic_inputs_shape.size());
            dynamic_inout_shape = &(dynamic_inputs_shape[static_cast<size_t>(group_index)]);
        } else if (io_type == "o") {
            GE_ASSERT_TRUE(static_cast<size_t>(group_index) < dynamic_outputs_shape.size(),
                "[Mobile] group_index is invaild, group_index: %d, dynamic_outputs_shape.size: %d",
                group_index, dynamic_outputs_shape.size());
            dynamic_inout_shape = &(dynamic_outputs_shape[static_cast<size_t>(group_index)]);
        } else {
            GELOGE(ge::FAILED, "[Mobile] unknown io type: %s.", io_type.c_str());
            return ge::FAILED;
        }
        GE_ASSERT_NOTNULL(dynamic_inout_shape, "[Mobile] dynamic inout shape is nullptr.");
        size_t i = 0;
        while (i < dynamic_inout_shape->size()) {
            (void)arg_info.arg_indexes.insert(arg_info.add + args_index++);
            i++;
        }
    }
    return ge::SUCCESS;
}

ge::Status FindArgInfosWithShapeInfoIndex(int32_t i,
    std::vector<KernelArgInfoWithShapeInfo>& arg_infos_with_shape_info,
    int32_t& arg_infos_with_shape_info_index)
{
    if (arg_infos_with_shape_info.size() > INT32_MAX) {
        GELOGE(ge::FAILED, "[Mobile] overflow, failed.");
        return ge::FAILED;
    }
    for (int32_t j = 0; j < static_cast<int32_t>(arg_infos_with_shape_info.size()); j++) {
        if (arg_infos_with_shape_info[static_cast<uint64_t>(j)].arg_indexes.find(i) != arg_infos_with_shape_info[static_cast<uint64_t>(j)].arg_indexes.end()) {
            arg_infos_with_shape_info_index = j;
            break;
        }
    }
    return ge::SUCCESS;
}

ge::Status SetArgsAndFlowTable(const domi::KernelDef& kernel, ge::mobile::proto::KernelDef* mobile_kernel,
    std::vector<KernelArgInfoWithShapeInfo>& arg_infos_with_shape_info,
    DynamicInputsOutputsShapeInfo& dynamic_inputsoutputs_shapeinfo, bool& is_flowtable)
{
    const uint64_t* args_data = reinterpret_cast<const uint64_t*>(kernel.args().data());
    std::vector<uint8_t> flowtable;
    std::vector<uint64_t> args(kernel.args_size() / static_cast<uint32_t>(PER_ARG_BYTES));
    int32_t args_index_new = 0;
    uint64_t offset = 0;
    int32_t group_index_pre = -1;
    std::string io_type_pre = "";
    for (uint32_t i = 0; i < kernel.args_size() / static_cast<uint32_t>(PER_ARG_BYTES); i++) {
        int32_t arg_infos_with_shape_info_index = -1;
        GE_ASSERT_TRUE(FindArgInfosWithShapeInfoIndex(static_cast<int32_t>(i), arg_infos_with_shape_info, arg_infos_with_shape_info_index) == ge::SUCCESS,
            "[Mobile] find arg infos with shape info index failed.");
        GELOGI("[Mobile] args index: %d, value: 0x%llx, arg_infos_with_shape_info_index: %d", i, *args_data, arg_infos_with_shape_info_index);
        if ((arg_infos_with_shape_info_index != -1) &&
            (arg_infos_with_shape_info[static_cast<uint64_t>(arg_infos_with_shape_info_index)].format == "desc")) {
            GELOGI("[Mobile] modify for secondary addr args.");
            GELOGI("[Mobile] arg_infos_with_shape_info: %s",
                arg_infos_with_shape_info[static_cast<uint64_t>(arg_infos_with_shape_info_index)].ToString().c_str());
            is_flowtable = true;
            const auto io_type = arg_infos_with_shape_info[static_cast<uint64_t>(arg_infos_with_shape_info_index)].type;
            const int32_t group_index = arg_infos_with_shape_info[static_cast<uint64_t>(arg_infos_with_shape_info_index)].group_index;
            if ((group_index_pre == group_index) && (io_type_pre == io_type)) {
                GELOGI("[Mobile] in same group, group_index_pre: %d, group_index: %d, io_type_pre: %s, io_type: %s.",
                    group_index_pre, group_index, io_type_pre.c_str(), io_type.c_str());
                args_data++;
                continue;
            }
            const uint64_t real_size = ConstructSecondaryAddrContext(flowtable,
                dynamic_inputsoutputs_shapeinfo.GetShapeInfo(io_type, static_cast<uint64_t>(group_index)));
            GELOGI("[Mobile] real_size: %d", real_size);
            args[args_index_new++] = 0x6B6B6B6B00000000UL | offset;
            offset += real_size;
            group_index_pre = group_index;
            io_type_pre = io_type;
        } else {
            GELOGI("[Mobile] copy for direct addr args.");
            args[args_index_new++] = *args_data;
        }
        args_data++;
    }
    GELOGI("[Mobile] args_index_new: %d", args_index_new);
    mobile_kernel->set_flowtable(static_cast<const void*>(flowtable.data()), flowtable.size());
    mobile_kernel->set_args(static_cast<const void*>(args.data()), static_cast<uint32_t>(args_index_new * PER_ARG_BYTES));
    mobile_kernel->set_args_size(static_cast<uint32_t>(args_index_new * PER_ARG_BYTES));
    return ge::SUCCESS;
}


ge::Status ConvertKernelDef(const ge::GeModelPtr& ge_model,
    const domi::KernelDef& kernel, ge::mobile::proto::KernelDef* mobile_kernel, bool& is_flowtable)
{
    mobile_kernel->set_stub_func(kernel.stub_func());
    mobile_kernel->set_block_dim(kernel.block_dim());

    // node dynamic inputs outputs
    std::vector<std::vector<ShapeInfo>> dynamic_inputs_shape;
    std::vector<std::vector<ShapeInfo>> dynamic_outputs_shape;
    const auto& graph = ge_model->GetGraph();
    for (const ge::NodePtr &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
        const std::string& node_name = node->GetName();
        if (kernel.stub_func().find(node_name) == std::string::npos) {
            continue;
        }
        GELOGI("[Mobile] process node: %s dynamic inputs and outputs.", node_name.c_str());
        std::vector<int64_t> input_offset;
        GE_ASSERT_TRUE(GetDynamicInputsInfo(node, input_offset, dynamic_inputs_shape) == ge::SUCCESS,
            "[Mobile] get dynamic inputs info failed.");
        std::vector<int64_t> output_offset;
        GE_ASSERT_TRUE(GetDynamicOutputsInfo(node, output_offset, dynamic_outputs_shape) == ge::SUCCESS,
            "[Mobile] get dynamic outputs info failed.");
        break;
    }

    // parse args_format
    std::string args_format = kernel.context().args_format();
    GELOGI("[Mobile] kernel.context.args_format: %s ", args_format.c_str());
    std::vector<KernelArgInfoWithShapeInfo> arg_infos_with_shape_info;
    GE_ASSERT_TRUE(ParseArgsFormat(args_format, arg_infos_with_shape_info) == ge::SUCCESS,
        "[Mobile] parse args format failed.");

    // set args info args index
    DynamicInputsOutputsShapeInfo dynamic_inputsoutputs_shapeinfo = {dynamic_inputs_shape, dynamic_outputs_shape};
    GE_ASSERT_TRUE(SetArgInfosArgsIndex(arg_infos_with_shape_info, dynamic_inputsoutputs_shapeinfo) == ge::SUCCESS,
        "[Mobile] set arg infos args index failed.");

    // set args and flowtable
    GE_ASSERT_TRUE(SetArgsAndFlowTable(kernel, mobile_kernel, arg_infos_with_shape_info,
        dynamic_inputsoutputs_shapeinfo, is_flowtable) == ge::SUCCESS, "[Mobile] get args and flowtable failed.");
    return ge::SUCCESS;
}

ge::Status ConvertTaskDef(const ge::GeModelPtr& ge_model,
    const std::shared_ptr<domi::ModelTaskDef>& model_task_def,
    std::shared_ptr<ge::mobile::proto::ModelTaskDef>& mobile_model_task_def)
{
    for (const auto& task: model_task_def->task()) {
        const auto& kernel = task.kernel();
        if (kernel.args_size() == 0) {
            GELOGI("[Mobile] skip kernel args size is 0");
            continue;
        }
        auto* mobile_task = mobile_model_task_def->add_task();
        mobile_task->set_id(task.id());
        mobile_task->set_type(static_cast<uint32_t>(RT_MODEL_TASK_KERNEL));
        mobile_task->set_stream_id(task.stream_id());
        GELOGI("[Mobile] stream_id: %d", static_cast<int32_t>(task.stream_id()));
        mobile_task->set_event_id(task.event_id());
        
        // KernelDef
        auto* mobile_kernel = mobile_task->mutable_kernel();
        bool is_flowtable = false;
        GE_ASSERT_TRUE(ConvertKernelDef(ge_model, kernel, mobile_kernel, is_flowtable) == ge::SUCCESS,
            "[Mobile] convert kernel def failed.");

        // KernelContext
        const auto& kernel_context = kernel.context();
        auto* mobile_kernel_context = mobile_kernel->mutable_context();
        mobile_kernel_context->set_kernel_type(static_cast<uint32_t>(ASCENDC_AI_CORE));
        mobile_kernel_context->set_kernel_func_id(kernel_context.kernel_func_id());
        GELOGI("[Mobile] kernel_func_id: %d", static_cast<int32_t>(kernel_context.kernel_func_id()));
        mobile_kernel_context->set_is_flowtable(is_flowtable);
        mobile_kernel_context->set_args_offset(kernel_context.args_offset());
        mobile_kernel_context->set_args_count(mobile_kernel->args_size() / static_cast<uint32_t>(PER_ARG_BYTES));

        // set args offset use args_count
        std::vector<char> args_offset(mobile_kernel_context->args_count() * sizeof(uint16_t), static_cast<char>(0));
        mobile_kernel_context->set_args_offset(args_offset.data(), args_offset.size());
        GELOGI("[Mobile] mobile args_count: %d", mobile_kernel_context->args_count());
    }
    return ge::SUCCESS;
}

ge::Status AddCompatibleInfoToCompiledTarget(ge::CompiledTargetPtr& compiled_target)
{
    ge::CompatibleInfoV2 compatible_info_v2;
    const size_t compatible_info_size = compatible_info_v2.GetSize();
    std::unique_ptr<uint8_t[]> compatible_info_data = std::make_unique<uint8_t[]>(compatible_info_size);
    GE_ASSERT_NOTNULL(compatible_info_data, "[Mobile] compatible_info_data is nullptr.");
    ge::BaseBuffer compatible_info_buffer(compatible_info_data.get(), compatible_info_size);
    const auto ret = compatible_info_v2.GetCompatibleInfo(compatible_info_buffer);
    GE_ASSERT_TRUE(ret == ge::SUCCESS, "[Mobile] GetCompatibleInfo failed.");
    GELOGI("[Mobile] compatible_info_buffer size: %d", compatible_info_buffer.GetSize());

    ge::SectionHolder compatible_info_section;
    compatible_info_section.type = static_cast<uint32_t>(ge::SectionType::SECTION_TYPE_COMPATIBLEV2);
    GE_ASSERT_TRUE(compatible_info_size <= UINT32_MAX, "[Mobile] overflow, failed.");
    compatible_info_section.size = static_cast<uint32_t>(compatible_info_size);
    compatible_info_section.data = std::move(compatible_info_data);
    compiled_target->AddSection(compatible_info_section);
    return ge::SUCCESS;
}

ge::Status AddKernelBinToManager(const ge::GeModelPtr& ge_model,
    ge::mobile::KernelBinManager& kernelbin_manager,
    std::shared_ptr<ge::mobile::proto::ModelTaskDef>& mobile_model_task_def)
{
    const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
    // -> <name, offset>
    const auto& graph = ge_model->GetGraph();
    GELOGI("[Mobile] convert kernel bin.");
    for (const ge::NodePtr& node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
        const std::string kernel_name_str = "_kernelname";
        std::string kernel_name = "";
        const string *kernel_name_ptr = ge::AttrUtils::GetStr(node->GetOpDesc(), kernel_name_str);
        if (kernel_name_ptr != nullptr) {
            kernel_name = *kernel_name_ptr;
        }
        const std::string& node_name = node->GetName();
        GELOGI("[Mobile] node name: %s, kernel name: %s", node_name.c_str(), kernel_name.c_str());
        auto kernel_bin = tbe_kernel_store.FindKernel(kernel_name);
        if (kernel_bin == nullptr) {
            GELOGI("[Mobile] not kernel bin find.");
            continue;
        }
        GELOGI("[Mobile] kernel bin data size: %d", kernel_bin->GetBinDataSize());
        ge::mobile::KernelBin mobile_kernel_bin;
        mobile_kernel_bin.kernel_info.func_mode = 0U;
        mobile_kernel_bin.kernel_info.magic = MOBILE_RT_DEV_BINARY_MAGIC_ELF;
        GE_ASSERT_TRUE(kernel_bin->GetBinDataSize() <= UINT32_MAX, "[Mobile] overflow, failed.");
        mobile_kernel_bin.kernel_info.kernel_size = static_cast<uint32_t>(kernel_bin->GetBinDataSize());
        mobile_kernel_bin.kernel_info.kernel_offset = 0U;
        mobile_kernel_bin.stub_func = kernel_bin->GetBinData();
        mobile_kernel_bin.stub_name = kernel_name;
        GELOGI("[Mobile] stub_name: %s", kernel_name.c_str());
        kernelbin_manager.AddKernelBin(mobile_kernel_bin);
        // replace task->kernel->stub_func -> kernel_name
        for (int i = 0; i < mobile_model_task_def->task_size(); i++) {
            auto* task = mobile_model_task_def->mutable_task(i);
            if (task->kernel().stub_func().find(node_name) != std::string::npos) {
                GELOGI("[Mobile] instead kernel stub func: %s  to: %s", task->kernel().stub_func().c_str(),
                    kernel_name.c_str());
                task->mutable_kernel()->set_stub_func(kernel_name);
            } else {
                GELOGI("[Mobile] can not find node name: %s in stub_func: %s", node_name.c_str(),
                    task->kernel().stub_func().c_str());
            }
        }
    }
    return ge::SUCCESS;
}

ge::Status AddKernelBinToCompiledTarget(ge::mobile::KernelBinManager& kernelbin_manager,
    ge::CompiledTargetPtr& compiled_target)
{
    const size_t kernelbin_section_size = kernelbin_manager.GetBinSectionSize();
    std::unique_ptr<uint8_t[]> kernelbin_section_data = std::make_unique<uint8_t[]>(kernelbin_section_size);
    GE_ASSERT_NOTNULL(kernelbin_section_data, "[Mobile] kernelbin_section_data is nullptr.");
    ge::BaseBuffer kernelbin_section_buffer(kernelbin_section_data.get(), kernelbin_section_size);
    const auto ret = kernelbin_manager.SaveKernelBinToBuffer(kernelbin_section_buffer);
    GE_ASSERT_TRUE(ret == ge::SUCCESS, "[Mobile] SaveKernelBinToBuffer failed.");
    GELOGI("[Mobile] kernelbin_section_buffer size: %d", kernelbin_section_buffer.GetSize());

    ge::SectionHolder kernelbins_section;
    kernelbins_section.type = static_cast<uint32_t>(ge::SectionType::SECTION_TYPE_BIN);
    GE_ASSERT_TRUE(kernelbin_section_size <= UINT32_MAX, "[Mobile] overflow, failed.");
    kernelbins_section.size = static_cast<uint32_t>(kernelbin_section_size);
    kernelbins_section.data = std::move(kernelbin_section_data);
    compiled_target->AddSection(kernelbins_section);
    return ge::SUCCESS;
}

ge::CompiledTargetPtr ConstructCompiledTarget(const ge::GeModelPtr& ge_model)
{
    ge::CompiledTargetPtr compiled_target = std::make_shared<ge::CompiledTarget>();

    // add model task def
    const std::shared_ptr<domi::ModelTaskDef> model_task_def = ge_model->GetModelTaskDefPtr();

    std::shared_ptr<ge::mobile::proto::ModelTaskDef> mobile_model_task_def =
        std::make_shared<ge::mobile::proto::ModelTaskDef>();

    const auto& model_task_def_attr = model_task_def->attr();
    for (const auto& it: model_task_def_attr) {
        GELOGI("[Mobile] attr %s : %s ", it.first.c_str(), it.second.c_str());
        (void)mobile_model_task_def->mutable_attr()->insert({it.first, it.second});
    }

    mobile_model_task_def->set_memory_size(model_task_def->memory_size());
    GELOGI("[Mobile] model task def memory size: %d", mobile_model_task_def->memory_size());
    GELOGI("[Mobile] stream num(ori): %d", static_cast<int32_t>(model_task_def->stream_num()));
    mobile_model_task_def->set_stream_num(1);
    GELOGI("[Mobile] stream num: %d", static_cast<int32_t>(mobile_model_task_def->stream_num()));
    mobile_model_task_def->set_event_num(model_task_def->event_num());
    mobile_model_task_def->set_weight_size(model_task_def->weight_size());

    // -> taskdef
    const auto ret = ConvertTaskDef(ge_model, model_task_def, mobile_model_task_def);
    GE_ASSERT_TRUE(ret == ge::SUCCESS, "[Mobile] convert task def failed.");

    // 2G size
    constexpr uint64_t data_mem_base = 2147483648;
    // set fm base addr
    mobile_model_task_def->set_base_addr(data_mem_base);
    // weight_addr should use data_mem_base + memory size(model task def)
    mobile_model_task_def->set_weight_addr(data_mem_base + mobile_model_task_def->memory_size());
    mobile_model_task_def->set_batch_num(model_task_def->batch_num());

    // add compatible info section
    GE_ASSERT_TRUE(AddCompatibleInfoToCompiledTarget(compiled_target) == ge::SUCCESS,
        "add compatible info to compiled target failed.");

    // add kernelbin section
    ge::mobile::KernelBinManager kernelbin_manager;
    GE_ASSERT_TRUE(AddKernelBinToManager(ge_model, kernelbin_manager,
        mobile_model_task_def) == ge::SUCCESS, "add kernel bin to manager failed.");
    compiled_target->AddModelTaskDef(mobile_model_task_def);
    GE_ASSERT_TRUE(AddKernelBinToCompiledTarget(kernelbin_manager,
        compiled_target) == ge::SUCCESS, "add kernel bin to manager failed.");
    return compiled_target;
}

ge::CompiledModelPtr ConstructCompiledModel(const ge::GeModelPtr& ge_model)
{
    auto compiled_model_ptr = std::make_shared<ge::CompiledModel>();
    GE_ASSERT_NOTNULL(compiled_model_ptr, "[Mobile] compiled_model_ptr is nullptr");

    // add ge model(MODEL_DEF)
    compiled_model_ptr->SetGeModel(ge_model);

    // add weights buffer(WEIGHTS_DATA)
    ge::BaseBuffer weight_buffer;
    weight_buffer.SetData(ge_model->GetWeightData());
    weight_buffer.SetSize(ge_model->GetWeightSize());
    compiled_model_ptr->AddWeight(weight_buffer);

    // add compiled targets(TASK_INFO)
    compiled_model_ptr->AddCompiledTarget(ConstructCompiledTarget(ge_model));
    return compiled_model_ptr;
}

} // namespace

namespace ge {

Status MobileModelHelper::SaveToOmRootModel(
    const GeRootModelPtr& ge_root_model,
    const std::string& output_file,
    ModelBufferData& model,
    const bool is_unknown_shape)
{
    GE_ASSERT_NOTNULL(ge_root_model, "[Mobile] ge_root_model is nullptr");
    GE_ASSERT_TRUE(!output_file.empty(), "[Mobile] SaveModel received invalid file name prefix.");
    const auto& name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
    GE_ASSERT_TRUE(!name_to_ge_model.empty(), "[Mobile] ge_root_model has no sub model.");
    if (!is_unknown_shape) {
        auto& model_root = name_to_ge_model.begin()->second;
        return SaveToOmModel(model_root, output_file, model, ge_root_model);
    }
    GELOGE(FAILED, "[Mobile] mobile is not support unknown shape model to om!!!");
    REPORT_INNER_ERR_MSG("E19999", "[Mobile] mobile is not support unknown shape model to om!!!");
    return FAILED;
}

Status MobileModelHelper::SaveToOmModel(
    const GeModelPtr& ge_model,
    const std::string& output_file,
    ModelBufferData& model,
    const GeRootModelPtr& ge_root_model)
{
    (void)ge_root_model;
    (void)model;
    GE_ASSERT_NOTNULL(ge_model, "ge_model is nullptr");
    if (output_file.empty()) {
        GELOGE(FAILED, "[Mobile] SaveModel received invalid file name prefix, "
            "model %s", ge_model->GetName().c_str());
        REPORT_INNER_ERR_MSG("E19999", "[Mobile] SaveModel received invalid file name prefix, "
                          "model %s", ge_model->GetName().c_str());
        return FAILED;
    }

    auto compiled_model_ptr = ConstructCompiledModel(ge_model);
    GE_ASSERT_NOTNULL(compiled_model_ptr, "[Mobile] compiled_model_ptr is nullptr");
    GELOGI("[Mobile] output file: %s", output_file.c_str());

    std::string external_weight_str = "0";
    (void)ge::GetContext().GetOption(ge::EXTERNAL_WEIGHT, external_weight_str);
    GELOGI("[Mobile] external weight str: %s", external_weight_str.c_str());

    std::string output_weight_dir = "";
    const auto pos = output_file.rfind("/");
    if (pos != std::string::npos) {
        output_weight_dir = output_file.substr(0, pos);
    }
    GELOGI("[Mobile] output weight dir: %s", output_weight_dir.c_str());

    ge::ModelFileSaver model_file_saver;
    const auto ret = model_file_saver.SaveCompiledModelToFile(
        compiled_model_ptr,
        (output_file + "c").c_str(),
        output_weight_dir.c_str(),
        external_weight_str == "1");

    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] model file saver, save failed.");
    return SUCCESS;
}

void MobileModelHelper::SetSaveMode(const bool val)
{
    is_offline_ = val;
}

REGISTER_MODEL_SAVE_HELPER(OM_FORMAT_MOBILE, MobileModelHelper);

}  // namespace ge

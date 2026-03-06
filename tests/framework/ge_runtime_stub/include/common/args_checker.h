/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_ARGS_CHECKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_ARGS_CHECKER_H_
#include <string>
#include <set>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include "framework/ge_runtime_stub/include/common/args_parser_factory.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "stub/gert_runtime_stub.h"
#include "graph/load/model_manager/davinci_model.h"
#include "session/session_manager.h"
#include "dflow/inc/data_flow/model/graph_model.h"

extern ge::SessionManager *GetSessionManager();

namespace ge {
struct AddrSegment {
  uint64_t addr;
  uint64_t length;
  int64_t offset;
};

class ArgsChecker {
 public:
  explicit ArgsChecker(ComputeGraphPtr graph, const uint32_t graph_id,
                       const uint64_t session_id, const gert::GertRuntimeStub &stub)
      : graph_(std::move(graph)),
        graph_id_(graph_id),
        session_id_(session_id),
        stub_(stub) {
    if (BuildOpIndexToArgsParserMap() != SUCCESS) {
      throw std::invalid_argument("build op index to task type map failed");
    }

    if (BuildOpNameToArgsAddrMap() != SUCCESS) {
      throw std::invalid_argument("build op name to args addr map failed");
    }
  }

  /**
   * 保存fm段实际地址和长度
   * @param addr
   * @param length
   * @return
   */
  Status SetFmAddr(const uint64_t addr, const uint64_t length) {
    AddrSegment addr_segment = {addr, length, 0};
    addr_segments_.emplace_back(addr_segment);
    GELOGD("set addr segment fm addr:%lu, length:%lu", addr, length);
    return SUCCESS;
  }

  /**
   * 保存model复用的input的的地址和长度
   * @param input_index
   * @param inputs
   * @return
   */
  template <typename T>
  Status SetModelInputAddr(const std::vector<int64_t> &input_index, const std::vector<T> &inputs) {
    for (const int64_t &index : input_index) {
      if (index >= (int64_t)inputs.size()) {
        GELOGE(FAILED, "index:%lu exceed inputs size:%z", index, inputs.size());
        return FAILED;
      }
      auto [addr, length] = ParseFromTensor(inputs[index]);

      uint32_t data_op_index = 0U;
      for (const auto &node : graph_->GetAllNodes()) {
        const auto &op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
          if (data_op_index == index) {
            const std::vector<int64_t> &output_offset = op_desc->GetOutputOffset();
            AddrSegment addr_segment = {addr, length, output_offset[0]};

            GELOGD("set addr segment data op name:%s, addr:%lu, length:%lu, offset:%ld",
              op_desc->GetName().c_str(), addr, length, output_offset[0]);

            addr_segments_.emplace_back(addr_segment);
            break;
          }
          data_op_index++;
        }
      }
    }
    return SUCCESS;
  }

  /**
   * 保存model复用的output的的地址和长度
   * @param output_index
   * @param outputs
   * @return
   */
  template <typename T>
  Status SetModelOutputAddr(const std::vector<int64_t> &output_index, const std::vector<T> &outputs) {
    for (const auto &node : graph_->GetAllNodes()) {
      const auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if (op_desc->GetType() == NETOUTPUT) {
        const std::vector<int64_t> &output_offset = op_desc->GetInputOffset();
        for (const int64_t &index : output_index) {
          if (index >= (int64_t)outputs.size()) {
             GELOGE(FAILED, "index:%lu exceed outputs size:%z", index, outputs.size());
            return FAILED;
          }
          auto [addr, length] = ParseFromTensor(outputs[index]);

          for (size_t i = 0; i < output_offset.size(); i++) {
            if (index == (int64_t)i) {
              AddrSegment addr_segment = {addr, length, output_offset[i]};
              GELOGD("set addr segment netoutput op name:%s, addr:%lu, length:%lu, offset:%ld",
                op_desc->GetName().c_str(), addr, length, output_offset[i]);

              addr_segments_.emplace_back(addr_segment);
              break;
            }
          }
        }
        break;
      }
    }
    return SUCCESS;
  }

  /**
   * 清除所有地址段信息
   * @param
   * @return
   */
  void ClearAddrSegments() {
    addr_segments_.clear();
  }

  /**
   * 校验地址是否正确
   * @param
   * @return
   */
  Status TaskIoAddressesAreCorrect() {
    if (AnchorAddressesWithSameSymbolAreConsistent() != SUCCESS) {
      return FAILED;
    }

    for (const auto &node : graph_->GetAllNodes()) {
      const auto &op_desc = node->GetOpDesc();
      GELOGD("op name:%s op type:%s libname:%s start to check.",
        op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetOpKernelLibName().c_str());

      if (op_desc->GetType() == VARIABLE) {
        GELOGD("op name :%s is variable.", op_desc->GetName().c_str());
        continue;
      }

      // 没有生成parser或args_addr的op过滤掉
      if (op_index_to_parser_.count(op_desc->GetId()) == 0 || op_name_to_args_addr_.count(op_desc->GetName()) == 0) {
        GELOGD("op name :%s no mathing task.", op_desc->GetName().c_str());
        continue;
      }

      uint64_t args_addr = op_name_to_args_addr_[op_desc->GetName()];
      GELOGD("op name:%s, args_addr:%lu.", op_desc->GetName().c_str(), args_addr);

      // 校验输入地址的offset
      if (CheckIoOffset(op_desc, args_addr, kIn) != SUCCESS) {
        return FAILED;
      }

      if (CheckIoOffset(op_desc, args_addr, kOut) != SUCCESS) {
        return FAILED;
      }

      if (!op_index_to_parser_[op_desc->GetId()]->CheckArgsExtra(args_addr)) {
        return FAILED;
      }

      GELOGD("op name:%s check end.", op_desc->GetName().c_str());
    }

    return SUCCESS;
  }

  /**
   * 校验指定节点发生了地址更新
   * @param node_names
   * @return
   */
  Status CheckNodesArgsUpdated(const std::vector<std::string> &node_names) {
    const GeFakeRtMemcpyArgs *rt_memcpy_args = nullptr;
    if (GetFakeRtMemcpyArgs(rt_memcpy_args) != SUCCESS) {
      return FAILED;
    }

    for (auto &node_name : node_names) {
      const auto node = graph_->FindNode(node_name);
      if (node == nullptr ||  node->GetOpDesc() == nullptr) {
        GELOGE(FAILED, "cannot find node name:%s", node_name.c_str());
        return FAILED;
      }

      const auto op_desc = node->GetOpDesc();
      uint64_t args_addr = op_name_to_args_addr_[op_desc->GetName()];
      uint64_t dst_address = PtrToValue(rt_memcpy_args->dst_address);
      if ((args_addr < dst_address) || (args_addr >= dst_address + rt_memcpy_args->copy_len)) {
        GELOGE(FAILED, "node name:%s, args addr:%lu mismatching copy dst addr:%lu, len:%u",
          node_name.c_str(), args_addr, dst_address,  rt_memcpy_args->copy_len);
        return FAILED;
      }

      GELOGD("node name:%s, args addr:%lu matching copy dst addr:%lu, len:%u",
        node_name.c_str(), args_addr, dst_address,  rt_memcpy_args->copy_len);
    }

    return SUCCESS;
  }

  /**
   * 校验指定节点未发生地址更新
   * @param node_names
   * @return
   */
  Status CheckNodesArgsNotUpdated(const std::vector<std::string> &node_names) {
    const GeFakeRtMemcpyArgs *rt_memcpy_args = nullptr;
    if (GetFakeRtMemcpyArgs(rt_memcpy_args) != SUCCESS) {
      return SUCCESS;
    }

    for (auto &node_name : node_names) {
      const auto node = graph_->FindNode(node_name);
      if (node == nullptr ||  node->GetOpDesc() == nullptr) {
        return FAILED;
      }

      const auto op_desc = node->GetOpDesc();
      uint64_t args_addr = op_name_to_args_addr_[op_desc->GetName()];
      uint64_t dst_address = PtrToValue(rt_memcpy_args->dst_address);
      if ((args_addr >= dst_address) && (args_addr < dst_address + rt_memcpy_args->copy_len)) {
        GELOGE(FAILED, "node name:%s, args addr:%lu matching copy dst addr:%lu, len:%u",
          node_name.c_str(), args_addr, dst_address, rt_memcpy_args->copy_len);
        return FAILED;
      }

      GELOGD("node name:%s, args addr:%lu mismatching copy dst addr:%lu, len:%u",
        node_name.c_str(), args_addr, dst_address,  rt_memcpy_args->copy_len);
    }

    return SUCCESS;
  }

private:
  /**
   * 构建opindex和task info的映射
   * @param
   * @return
   */
  Status BuildOpIndexToArgsParserMap() {
    SessionManager *session_manager = GetSessionManager();
    GE_CHECK_NOTNULL(session_manager);

    SessionPtr session = session_manager->GetSession(session_id_);
    GE_CHECK_NOTNULL(session);

    const GraphManager &graph_manager = session->getGraphManagerObj(); // 当前无函数可以获取graph manager
    GraphNodePtr graph_node;
    Status ret = graph_manager.GetGraphNode(graph_id_, graph_node);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "get graph node failed, graph id:%u", graph_id_);
      return FAILED;
    }
    const auto ge_root_model = graph_node->GetGeRootModel();
    GE_CHECK_NOTNULL(ge_root_model);

    const auto &root_graph = ge_root_model->GetRootGraph();
    GE_CHECK_NOTNULL(root_graph);

    const auto &name_to_model = ge_root_model->GetSubgraphInstanceNameToModel();
    const auto it = name_to_model.find(root_graph->GetName());
    const GeModelPtr ge_model = (it != name_to_model.end()) ? it->second : nullptr;
    GE_CHECK_NOTNULL(ge_model);

    std::unordered_map<int64_t, OpDescPtr> index_to_desc;
    for (const auto &node : graph_->GetAllNodes()) {
      const auto &op_desc = node->GetOpDesc();
      index_to_desc.emplace(op_desc->GetId(), op_desc);
    }

    const auto &model_task_def = ge_model->GetModelTaskDefPtr();
    const size_t task_size = static_cast<size_t>(model_task_def->task_size());
    for (size_t i = 0; i < task_size; ++i) {
      const auto &task_def = model_task_def->task(i);
      const auto &task_info = TaskInfoFactory::Instance().Create(
          static_cast<ModelTaskType>(task_def.type()));
      GE_CHECK_NOTNULL(task_info);
      const auto op_index = task_info->ParseOpIndex(task_def);
      auto parser = ArgsParserFactory::CreateBy(task_def, std::move(index_to_desc[op_index]));
      if (parser == nullptr) {
        GELOGD("Task of type[%u] not yet supported in ArgsChecker", task_def.type());
        continue;
      }
      op_index_to_parser_[op_index] = std::move(parser);
    }

    return SUCCESS;
  }

  Status rtSwitchArgsTraversal(uint64_t &op_args_addr_min, uint64_t &op_args_addr_max) {
    for (const auto &switch_arg : stub_.GetAclRuntimeStub().GetAllSwitchArgs()) {
      auto node_name = switch_arg.GetTag();
      uint64_t addresses = 0;
      if (node_name == nullptr) {
        GELOGD("switch without a tag.");
        continue;
      }

      addresses = switch_arg.switch_addr_;
      if (addresses == 0) {
          GELOGE(FAILED, "switch node:%s addr is 0.", (*node_name).c_str(), addresses);
          return FAILED;
      }

      const auto node = graph_->FindNode(*node_name);
      if (node == nullptr ||  node->GetOpDesc() == nullptr) {
        GELOGE(FAILED, "switch node:%s cannot find address.", (*node_name).c_str());
        return FAILED;
      }

      std::string node_name_str = *node_name;
      op_name_to_args_addr_[node_name_str] = addresses;
    }

    GELOGD("op args addr min:%lu, addr max:%lu.", op_args_addr_min, op_args_addr_max);
    return SUCCESS;
  }

  Status rtLaunchArgsTraversal(uint64_t &op_args_addr_min, uint64_t &op_args_addr_max) {
    for (const auto &launch_arg : stub_.GetRtsRuntimeStub().GetAllLaunchArgs()) {
      auto node_name = launch_arg.GetTag();
      uint64_t addresses = 0;
      if (node_name == nullptr) {
        GELOGD("launch without a tag.");
        continue;
      }

      addresses = PtrToValue(launch_arg.args_addr_);
      if (addresses == 0) {
          GELOGE(FAILED, "launch node:%s addr is 0.", (*node_name).c_str(), addresses);
          return FAILED;
      }

      const auto node = graph_->FindNode(*node_name);
      if (node == nullptr ||  node->GetOpDesc() == nullptr) {
        GELOGE(FAILED, "launch node:%s cannot find address.", (*node_name).c_str());
        return FAILED;
      }

      GELOGD("launch node:%s addr:%lu, stream id:%u, task id:%u.",
        (*node_name).c_str(), addresses, launch_arg.GetStreamId(), launch_arg.GetTaskId());

      std::string node_name_str = *node_name;
      op_name_to_args_addr_[node_name_str] = addresses;

      // todo：MemcpyAsync为条件算子或者dsa插入的op, st使用ts内存, 不在hbm里
      if (node_name_str.find("MemcpyAsync") == std::string::npos) {
        op_args_addr_min =
          op_args_addr_min == 0 ? addresses : (addresses < op_args_addr_min ? addresses : op_args_addr_min);
        op_args_addr_max =
          op_args_addr_max == 0 ? addresses : (addresses > op_args_addr_max ? addresses : op_args_addr_max);
      }
    }
    GELOGD("op args addr min:%lu, addr max:%lu.", op_args_addr_min, op_args_addr_max);
    return SUCCESS;
  }

  /**
   * 构建opname和args addr的映射
   * @param
   * @return
   */
  Status BuildOpNameToArgsAddrMap() {
    op_args_addr_min_ = 0;
    op_args_addr_max_ = 0;
    
    auto ret = rtSwitchArgsTraversal(op_args_addr_min_, op_args_addr_max_);
    if (ret != SUCCESS) {
      return ret;
    }
  
    ret = rtLaunchArgsTraversal(op_args_addr_min_, op_args_addr_max_);

    return ret;
  }


  /**
   * 校验同一symbol下所有anchor的地址是否一致
   * @param
   * @return
   */
  // todo swtich的const 输出和其添加的rtmempcpy的地址不一致
  Status AnchorAddressesWithSameSymbolAreConsistent() {
    SymbolToAnchors symbol_to_anchors;
    AnchorToSymbol anchor_to_symbol;

    if (GraphUtils::GetRefMapping(graph_, symbol_to_anchors, anchor_to_symbol) != SUCCESS) {
      return FAILED;
    }

    for (const auto &[symbol, anchors] : symbol_to_anchors) {
      uint64_t addr_reference = 0;
      for (const auto &node_index_io : anchors) {
        std::string node_name  = node_index_io.node_ptr_->GetName();
        std::string op_desc_name = node_index_io.node_ptr_->GetOpDesc()->GetName();
        int64_t op_index = node_index_io.node_ptr_->GetOpDesc()->GetId();
        IOType io_type = node_index_io.io_type_;
        uint32_t io_index =  node_index_io.index_;
        uint64_t addr = 0;

        GELOGD("symbol:%s, node name:%s, op index:%ld, opdesc name:%s, io index:%u, io type:%d",
          symbol.c_str(), node_name.c_str(), op_index, op_desc_name.c_str(), io_index, io_type);

        // data op的input offset以及netoutput的outoffset 为0，过滤掉
        if ((io_type == kIn && node_index_io.node_ptr_->GetOpDesc()->GetInputOffset().size() == 0) ||
          (io_type == kOut && node_index_io.node_ptr_->GetOpDesc()->GetOutputOffset().size() == 0)) {
          GELOGD("symbol:%s, node name:%s, op index:%ld, opdesc name:%s has no offset",
            symbol.c_str(), node_name.c_str(), op_index, op_desc_name.c_str());
          continue;
        }

        // 没有生成parser或args_addr的op过滤掉
        if (op_index_to_parser_.count(op_index) == 0 || op_name_to_args_addr_.count(op_desc_name) == 0) {
          GELOGD("symbol:%s, op name :%s no mathing task.", symbol.c_str(), op_desc_name.c_str());
          continue;
        }

        uint64_t args_addr = op_name_to_args_addr_[op_desc_name];
        const auto &parser = op_index_to_parser_[op_index];
        addr = parser->ParseArgsAddr(args_addr, io_type == kIn, io_index);

        // todo st用例中，data op在args table中输出填入的地址为0，过滤掉； dsa 的input2/input3未解析，过滤掉
        if (addr == 0) {
            GELOGD("symbol:%s, node name:%s, op index:%ld, io index:%u, io type:%d, addr:%lu",
              symbol.c_str(), node_name.c_str(), op_index, io_index, io_type, addr);
            continue;
        }

        if (addr_reference == 0) {
          addr_reference = addr;
        }

        if (addr_reference != addr) {
          GELOGE(FAILED, "symbol:%s, node name:%s, op index:%ld, " \
            "io index:%u, io type:%d, addr_reference:%lu, addr:%lu, addr check failed.",
            symbol.c_str(), node_name.c_str(), op_index, io_index, io_type, addr_reference, addr);
          return FAILED;
        }

        GELOGD("symbol:%s, node name:%s, op index:%ld, " \
          "io index:%u, io type:%d, addr_reference:%lu, addr:%lu, addr check success",
          symbol.c_str(), node_name.c_str(), op_index, io_index, io_type, addr_reference, addr);
      }
    }

    return SUCCESS;
  }


  /**
   * 判断实际IO地址推算的offset和编译态生成的opdesc里的offset是否一致
   * @param op_desc
   * @param args_addr
   * @param io_type
   * @return
   */
  Status CheckIoOffset(const OpDescPtr op_desc, const uint64_t args_addr, const IOType io_type) {
    std::vector<int64_t> offset = (io_type == kIn ? op_desc->GetInputOffset() : op_desc->GetOutputOffset());
    if (offset.size() == 0) {
      GELOGD("op name:%s input offset size is 0.", op_desc->GetName().c_str());
      return SUCCESS;
    }

    const auto &parser = op_index_to_parser_[op_desc->GetId()];
    int32_t addr_segments_match = 0;
    for (size_t i = 0; i < offset.size(); i++) {
      uint64_t addr = parser->ParseArgsAddr(args_addr, io_type == kIn, i);

      for (size_t j = 0; j < addr_segments_.size(); j++) {
        if (addr >= addr_segments_[j].addr && addr < addr_segments_[j].addr + addr_segments_[j].length) {
          addr_segments_match = 1;
          // 根据实际地址 - 段实际起始地址 + 段的逻辑偏移，推算逻辑offset
          uint64_t addr_offset = addr -  addr_segments_[j].addr + addr_segments_[j].offset;
          if (addr_offset != (uint64_t)offset[i]) {
            GELOGE(FAILED, "node name:%s, op index:%ld, io index:%u, io type:%d, addr:%lu, offset:%lu, " \
              "is inconsistent with compiled offset:%lu",
              op_desc->GetName().c_str(), op_desc->GetId(), i, io_type, addr, addr_offset, offset[i]);
            return FAILED;
          } else {
            GELOGD("node name:%s, op index:%ld, io index:%u, io type:%d, addr:%lu, offset:%lu, " \
              "is consistent with compiled offset:%lu",
              op_desc->GetName().c_str(), op_desc->GetId(), i, io_type, addr, addr_offset, offset[i]);
          }
        }
      }

      if (addr_segments_match == 0) {
        // dt中data的output addr为0
        // 非fm和io段的地址，比如fix地址，不在addr segments内
        // todo ：增加地址是否在fm和io段内校验的公共接口， 用例里面新增该校验点
        GELOGD("op name:%s io type:%d, addr:%lu is no matching addr segment.",
            op_desc->GetName().c_str(), io_type, addr);
      }
    }

    return SUCCESS;
  }

  /**
   * 获取args table H2D拷贝的fake信息
   * @param op_desc
   * @return
   */
  Status GetFakeRtMemcpyArgs(const GeFakeRtMemcpyArgs *&rt_memcpy_args) {
    for (const auto &args : stub_.GetAclRuntimeStub().GetRtMemcpyRecords()) {
      uint64_t dst_address = PtrToValue(args.dst_address);
      if (dst_address >= op_args_addr_min_ && dst_address <= op_args_addr_max_) {
        rt_memcpy_args = &args;
        GELOGD("rt memcpy args addr:%lu, len:%u",
          PtrToValue(rt_memcpy_args->dst_address), rt_memcpy_args->copy_len);
        return SUCCESS;
      }
    }

    GELOGD("rt memcpy args cannot find");
    return FAILED;
  }

  static std::pair<uint64_t, uint64_t> ParseFromTensor(const ge::Tensor &tensor) {
    return std::make_pair(PtrToValue(tensor.GetData()), tensor.GetSize());
  }
  static std::pair<uint64_t, uint64_t> ParseFromTensor(const gert::Tensor &tensor) {
    return std::make_pair(PtrToValue(tensor.GetAddr()), tensor.GetSize());
  }

 private:
  ComputeGraphPtr graph_;
  uint32_t graph_id_;
  uint64_t session_id_;
  const gert::GertRuntimeStub &stub_;
  std::vector<AddrSegment> addr_segments_;
  std::unordered_map<std::string, uint64_t> op_name_to_args_addr_;
  uint64_t op_args_addr_min_;
  uint64_t op_args_addr_max_;
  std::unordered_map<int64_t, std::unique_ptr<ArgsParser>> op_index_to_parser_;
};
} // namespace ge
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_ARGS_CHECKER_H_

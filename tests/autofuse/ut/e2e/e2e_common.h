/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "schedule_result.h"
#include "ascendc_ir.h"

void InitScheduleResultsByImplGraphs(const std::vector<ge::AscGraph> &impl_graphs,
                                     ascir::FusedScheduledResult &fused_schedule_result) {
  for (auto node : impl_graphs[0].GetAllNodes()) {
    if (node->GetType() == "Data") {
      int64_t index = -1;
      if (node->attr.ir_attr->GetAttrValue("index", index) == ge::GRAPH_FAILED) {
        auto &attr = reinterpret_cast<ge::AscDataIrAttrDef &>(*node->attr.ir_attr);
        attr.SetIndex(static_cast<int64_t>(fused_schedule_result.input_nodes.size()));
      }
      fused_schedule_result.input_nodes.emplace_back(node);
    } else if (node->GetType() == "Output") {
      int64_t index = -1;
      if (node->attr.ir_attr->GetAttrValue("index", index) == ge::GRAPH_FAILED) {
        auto &attr = reinterpret_cast<ge::AscDataIrAttrDef &>(*node->attr.ir_attr);
        attr.SetIndex(static_cast<int64_t>(fused_schedule_result.output_nodes.size()));
      }
      fused_schedule_result.output_nodes.emplace_back(node);
    } else if (node->GetType() == "Workspace") {
      fused_schedule_result.workspace_nodes.emplace_back(node);
    }
  }
  for (auto& schedule_results: fused_schedule_result.node_idx_to_scheduled_results) {
    for (auto& schedule_result: schedule_results) {
      schedule_result.schedule_groups.emplace_back(ascir::ScheduleGroup{impl_graphs});
    }
  }
}


void AssignDefaultIoIndex(ge::AscGraph &graph) {
  int32_t data_index = 0;
  int32_t output_index = 0;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Data") {
      auto &attr = reinterpret_cast<ge::AscDataIrAttrDef &>(*node->attr.ir_attr);
      attr.SetIndex(data_index++);
    } else if (node->GetType() == "Output") {
      node->attr.ir_attr = ge::AscDataIrAttrDef().Clone();
      auto &attr = reinterpret_cast<ge::AscDataIrAttrDef &>(*node->attr.ir_attr);
      attr.SetIndex(output_index++);
    } else {
    }
  }
}

std::string OptilingStub() {
  std::stringstream ss;
	ss << "#include <iostream>" << std::endl;
	ss << "#include <fstream>" << std::endl;
	ss << "#include <cinttypes>" << std::endl;
	ss << "#include <sys/syscall.h>" << std::endl;
	ss << "#include <unistd.h>" << std::endl;
	ss << "#include \"toolchain/slog.h\"" << std::endl;
	ss << "#define OP_LOGD(name, fmt, ...)" << std::endl;
	ss << "#define OP_LOGI(name, fmt, ...)" << std::endl;
	ss << "#define GE_MODULE_NAME static_cast<int32_t>(45)" << std::endl;
	ss << "inline uint64_t GetTid() {" << std::endl;
	ss << "     return static_cast<uint64_t>(syscall(__NR_gettid));" << std::endl;
	ss << "}" << std::endl;

	ss << "#define GELOGE(ERROR_CODE, fmt, ...)" << "\\" << std::endl;
	ss << " do {" << "\\" << std::endl;
	ss << "     dlog_error(GE_MODULE_NAME, \"%\" PRIu64 \" %s:ErrorNo: %\" PRIuLEAST8 \"(%s) %s\" fmt, " << "\\" << std::endl;
	ss << "                GetTid(), &__FUNCTION__[0U], (ERROR_CODE), \"\", \"\", ##__VA_ARGS__);" << "\\" << std::endl;
	ss << " } while (false)" << std::endl;

	ss << "#define OP_LOGE(name, fmt, ...) GELOGE(-1, \"[%s]\" fmt, name, ##__VA_ARGS__)" << std::endl;
	ss << "#define OP_NAME \"asc0000_autofused_abs\"" << std::endl;
	ss << "namespace optiling {" << std::endl;
	ss << "static bool GetTiling(AutofuseTilingData &tiling_data, int32_t tilingCaseId) {" << std::endl;
	ss << "  (void)tilingCaseId;" << std::endl;
	ss << "  return true;" << std::endl;
	ss << "}" << std::endl;
  ss << "using namespace std;" << std::endl;
  ss << "inline bool IsEqual(double a, double b) {return true;}" << std::endl;
	ss << "}" << std::endl;

  return ss.str();
}

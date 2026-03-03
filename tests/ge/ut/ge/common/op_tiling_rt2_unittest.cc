/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "graph/ge_tensor.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/anchor_utils.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "op_tiling/op_compile_info_manager.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "common/op_tiling/tiling_memcheck.h"
#include "common/op_tiling/tiling_dfx.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/atomic_clean_tiling_context.h"
#include "common/share_graph.h"
#include "common/env_path.h"

#include "common/sgt_slice_type.h"
#include "faker/space_registry_faker.h"

using namespace std;
using namespace ge;
using namespace gert;

namespace optiling {
namespace {
static string parse_int(const std::string& data) {
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }
  return result;
}
}
class CompileInfoJson : public CompileInfoBase {
public:
  CompileInfoJson(const std::string &json) : json_str_(json) {}
  ~CompileInfoJson() {}
private:
  std::string json_str_;
};

class RegisterOpTilingRT2UT : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};
uint32_t tiling_parse_count = 0;
struct DummyTilingParams {
  int64_t x;
  int64_t y;
};
struct DummyCompileInfo {
  int64_t tiling_key;
  int64_t block_dim;
  bool is_need_atomic;
  int64_t tiling_cond;
  std::vector<int64_t> c;
};
struct DynamicAtomicAddrCleanCompileInfo : public optiling::CompileInfoBase {
  uint32_t workspace_num = 0;
  uint32_t core_num = 0;
  uint32_t ub_size = 0;
  std::vector<int64_t> _workspace_index_list;
};
struct DynamicAtomicAddrCleanParam {
  uint32_t workspace_num = 0;
  uint32_t core_num = 0;
  uint32_t ub_size = 0;
  std::vector<int64_t> clean_workspace_size;  // write input to tiling data
  std::vector<int64_t> clean_output_size; // write input to tiling data
};
ge::graphStatus DummyTiling(TilingContext *tiling_context) {
  auto compile_info = reinterpret_cast<const DummyCompileInfo *>(tiling_context->GetCompileInfo());
  // simulate op write tiling info
  tiling_context->SetTilingKey(compile_info->tiling_key);
  tiling_context->SetBlockDim(compile_info->block_dim);
  tiling_context->SetNeedAtomic(compile_info->is_need_atomic);
  tiling_context->SetTilingCond(compile_info->tiling_cond);
  // write tiling data
  DummyTilingParams *tiling_data_ptr = tiling_context->GetTilingData<DummyTilingParams>();
  EXPECT_NE(tiling_data_ptr, nullptr);
  tiling_data_ptr->x = 1;
  tiling_data_ptr->y = 2;

  // write workspace {0,1,2}
  for (size_t i = 0U; i < 3; ++i) {
    size_t *workspace_size = tiling_context->GetWorkspaceSizes(i + 1);
    *(workspace_size + i) = i;
  }

  //  强一致性计算紧急需求上库，ge暂时不能依赖metadef，已于BBIT及本地验证DT通过，后续补上
  //  auto deterministic_level = tiling_context->GetDeterministicLevel();
  //  EXPECT_EQ(deterministic_level, 0);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DummyTilingParse(TilingParseContext *tiling_parse_context) {
  tiling_parse_count++;
  auto compile_info = tiling_parse_context->GetCompiledInfo<DummyCompileInfo>();
  compile_info->tiling_key = 123;
  compile_info->block_dim = 456;
  compile_info->is_need_atomic = true;
  compile_info->tiling_cond = 789;
  return ge::GRAPH_SUCCESS;
}

UINT32 MemsetTilingParse(gert::KernelContext *kernel_context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(ConcatV2).TilingParse<DummyCompileInfo>(DummyTilingParse).Tiling(DummyTiling);

IMPL_OP(Batch).TilingParse<DummyCompileInfo>(DummyTilingParse).Tiling(DummyTiling);

IMPL_OP(IncreFlashAttention).TilingParse<DummyCompileInfo>(DummyTilingParse).Tiling(DummyTiling);

IMPL_OP(CTCBeamSearchDecoder).TilingParse<DummyCompileInfo>(DummyTilingParse).Tiling(DummyTiling);

IMPL_OP(GroupedMatMulAllReduce).TilingParse<DummyCompileInfo>(DummyTilingParse).Tiling(DummyTiling);

// 把输入的compile info写入out workspace，用于外部校验vn
ge::graphStatus DummyTilingForDynamicAtomicAddrClean(TilingContext *context) {
  auto compute_node_info = context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto compile_info = reinterpret_cast<const DynamicAtomicAddrCleanCompileInfo *>(context->GetCompileInfo());
  if (compile_info == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const std::vector<int64_t> &workspace_idx = compile_info->_workspace_index_list;

  size_t clean_tensor_num = compute_node_info->GetInputsNum() - 1;

  TilingData *tiling_data = context->GetRawTilingData();
  if (tiling_data == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op: tiling_data nullptr!");
    return ge::GRAPH_FAILED;
  }

  auto atomic_clean_context = reinterpret_cast<AtomicCleanTilingContext *>(context);
   // write workspace
  size_t *workspace_size = context->GetWorkspaceSizes(6);
  *workspace_size = compile_info->core_num;
  *(workspace_size + 1) = compile_info->ub_size;
  *(workspace_size + 2) = compile_info->workspace_num;
  if (!workspace_idx.empty()) {
    *(workspace_size + 3) = workspace_idx[0];
  }

  size_t idx = 0U;
  for (; idx < clean_tensor_num; ++idx) {
    auto tensor_size = atomic_clean_context->GetCleanOutputSize(idx);
    *(workspace_size + 4 + idx) = tensor_size;
  }

  if (!workspace_idx.empty()) {
    auto ws_sizes = atomic_clean_context->GetCleanWorkspaceSizes();
    if (ws_sizes == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "op: ws_size nullptr!");
      return ge::GRAPH_FAILED;
    }
    if (ws_sizes->GetSize() == 0U) {
      GELOGE(ge::GRAPH_FAILED, "Clean workspace size is 0!");
      return ge::GRAPH_FAILED;
    }
    auto ws_size_data = reinterpret_cast<const uint64_t *>(ws_sizes->GetData());
    for (size_t i = 0U; i < workspace_idx.size(); ++i, ++idx) {
      auto tensor_size = ws_size_data[workspace_idx[i]];
      *(workspace_size + 4 + idx) = tensor_size;
    }
  }
  return ge::GRAPH_SUCCESS;
}

// 校验同一个compile info进来， tiling parse的缓存功能
TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingSuccessTwice) {
  tiling_parse_count = 0U;
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::ConcatV2ConstDependencyGraph();
  auto concatv2_node = graph->FindNode("concatv2");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(tiling_parse_count, 1U);
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ParseContextHolderPtr default_context_holder;
  EXPECT_NE(op_desc->TryGetExtAttr(OP_TILING_PARSE_RESULT, default_context_holder), nullptr);

  // check tiling output
  EXPECT_EQ(run_info.GetTilingKey(), 123);
  EXPECT_EQ(run_info.GetBlockDim(), 456);
  EXPECT_EQ(run_info.GetClearAtomic(), true);
  EXPECT_EQ(run_info.GetTilingCond(), 789);
  std::string tiling_data_str = parse_int(run_info.GetAllTilingData().str());
  EXPECT_EQ(tiling_data_str, "1 2 ");
  auto workspace = run_info.GetAllWorkspaces();
  EXPECT_EQ(run_info.GetWorkspaceNum(), 3);
  EXPECT_EQ(workspace[0], 0);
  EXPECT_EQ(workspace[1], 1);
  EXPECT_EQ(workspace[2], 2);

  utils::OpRunInfo run_info2;
  ret = AicoreRtParseAndTiling(op, platform_infos, run_info2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(tiling_parse_count, 1U);
   // check tiling output
  EXPECT_EQ(run_info2.GetTilingKey(), 123);
  EXPECT_EQ(run_info2.GetBlockDim(), 456);
  EXPECT_EQ(run_info2.GetClearAtomic(), true);
  EXPECT_EQ(run_info2.GetTilingCond(), 789);
  tiling_data_str = parse_int(run_info2.GetAllTilingData().str());
  EXPECT_EQ(tiling_data_str, "1 2 ");
  workspace = run_info2.GetAllWorkspaces();
  EXPECT_EQ(run_info2.GetWorkspaceNum(), 3);
  EXPECT_EQ(workspace[0], 0);
  EXPECT_EQ(workspace[1], 1);
  EXPECT_EQ(workspace[2], 2);
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingWithOpCoreNumSuccess) {
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  tiling_parse_count = 0U;
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::ConcatV2ConstDependencyGraph();
  auto concatv2_node = graph->FindNode("concatv2");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "_op_aicore_num", "5");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "_op_vectorcore_num", "10");
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(tiling_parse_count, 1U);
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ParseContextHolderPtr default_context_holder;
  EXPECT_NE(op_desc->TryGetExtAttr(OP_TILING_PARSE_RESULT, default_context_holder), nullptr);

  // check tiling output
  EXPECT_EQ(run_info.GetTilingKey(), 123);
  EXPECT_EQ(run_info.GetBlockDim(), 456);
  EXPECT_EQ(run_info.GetClearAtomic(), true);
  EXPECT_EQ(run_info.GetTilingCond(), 789);
  std::string tiling_data_str = parse_int(run_info.GetAllTilingData().str());
  EXPECT_EQ(tiling_data_str, "1 2 ");
  auto workspace = run_info.GetAllWorkspaces();
  EXPECT_EQ(run_info.GetWorkspaceNum(), 3);
  EXPECT_EQ(workspace[0], 0);
  EXPECT_EQ(workspace[1], 1);
  EXPECT_EQ(workspace[2], 2);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingWithOpCoreNumInvalid) {
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  tiling_parse_count = 0U;
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::ConcatV2ConstDependencyGraph();
  auto concatv2_node = graph->FindNode("concatv2");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "_op_aicore_num", "5");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "_op_vectorcore_num", "bb");
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_NE(ret, GRAPH_SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingWithOpCoreNumInvalid2) {
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  tiling_parse_count = 0U;
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::ConcatV2ConstDependencyGraph();
  auto concatv2_node = graph->FindNode("concatv2");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "_op_aicore_num", "100");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "_op_vectorcore_num", "bb");
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_NE(ret, GRAPH_SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

// 自动融合算子tiling ut验证
TEST_F(RegisterOpTilingRT2UT, AutofuseNodeAicoreParseAndTilingSuccess) {
  auto graph = ShareGraph::AutoFuseNodeGraph();
  auto autofuse_node = graph->FindNode("fused_graph");
  std::string cmake_binary_path = CMAKE_BINARY_DIR;
  auto autofuse_stub_so = cmake_binary_path + "/tests/depends/op_stub/libautofuse_stub.so";

  (void)ge::AttrUtils::SetStr(autofuse_node->GetOpDesc(), "bin_file_path", autofuse_stub_so);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(autofuse_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  // check tiling output
  EXPECT_EQ(run_info.GetTilingKey(), 0);
  EXPECT_EQ(run_info.GetBlockDim(), 8);
  std::string tiling_data_str = parse_int(run_info.GetAllTilingData().str());
  EXPECT_EQ(tiling_data_str, "1 2 3 ");
  auto workspace = run_info.GetAllWorkspaces();
  EXPECT_EQ(run_info.GetWorkspaceNum(), 1);
  EXPECT_EQ(workspace[0], 1024);
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckDynamicInputDescSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::IFASingleGraph();
  auto ifa_node = graph->FindNode("IncreFlashAttention");
  ge::AttrUtils::SetStr(ifa_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(ifa_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(ifa_node->GetOpDesc(), "ori_op_para_size", 24);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(ifa_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // check tiling output
  std::vector<ge::ArgDesc> arg_descs;
  size_t arg_id = 0;
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, arg_id++);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_DESC, arg_id++, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_DESC, arg_id++, true);
  for (size_t i = 0; i < 12UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, arg_id++);
  }
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 0);
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(ifa_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 24 112 112 0 0 4 0 0 0 0 0 24 24 0 0 24 0 1 2 176 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckOptionalInputNoPlaceholderSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::IFASingleGraph();
  auto ifa_node = graph->FindNode("IncreFlashAttention");
  ge::AttrUtils::SetStr(ifa_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(ifa_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(ifa_node->GetOpDesc(), "ori_op_para_size", 24);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(ifa_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  std::vector<ge::ArgDesc> arg_descs;
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, 0);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_DESC, 1, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_DESC, 2, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, 5);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, 11);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, 12);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 0);
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(ifa_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 24 112 112 4 24 24 24 0 1 2 104 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckDynamicInputEmptySuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::GroupedMatMulAllReduceSingleGraph();
  auto matmul_node = graph->FindNode("GroupedMatMulAllReduce");
  ge::AttrUtils::SetStr(matmul_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(matmul_node->GetOpDesc(), "_memcheck", true);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(matmul_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::ArgDesc> arg_descs;
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_DESC, 0, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_DESC, 1, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, 3);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_DESC, 0, true);
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(matmul_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "112 112 4 112 0 1 2 72 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckDynamicOutputEmptySuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::CTCBeamSearchDecoderSingleGraph();
  auto ctc_node = graph->FindNode("CTCBeamSearchDecoder");
  ge::AttrUtils::SetStr(ctc_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(ctc_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(ctc_node->GetOpDesc(), "ori_op_para_size", 24);
  (void)ge::AttrUtils::SetInt(ctc_node->GetOpDesc(), "op_para_size", 1000);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(ctc_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  std::vector<ge::ArgDesc> arg_descs;
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, 0);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, 1);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_DESC, 0, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_DESC, 2, true);
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(ctc_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 24 24 112 112 0 1 2 80 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckEmptyArgsformat1) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::CTCBeamSearchDecoderSingleGraph();
  auto ctc_node = graph->FindNode("CTCBeamSearchDecoder");
  ge::AttrUtils::SetStr(ctc_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(ctc_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(ctc_node->GetOpDesc(), "ori_op_para_size", 24);
  (void)ge::AttrUtils::SetInt(ctc_node->GetOpDesc(), "op_para_size", 1000);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(ctc_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  std::vector<ge::ArgDesc> arg_descs;
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(ctc_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 24 24 24 24 24 24 0 1 2 96 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckEmptyArgsformat2) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::IFASingleGraph();
  auto ifa_node = graph->FindNode("IncreFlashAttention");
  ge::AttrUtils::SetStr(ifa_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(ifa_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(ifa_node->GetOpDesc(), "ori_op_para_size", 24);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(ifa_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  std::vector<ge::ArgDesc> arg_descs;
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(ifa_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 24 24 24 24 24 24 24 4 24 24 24 0 1 2 136 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckDynamicInputSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::IFASingleGraph();
  auto ifa_node = graph->FindNode("IncreFlashAttention");
  ge::AttrUtils::SetStr(ifa_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(ifa_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetStr(ifa_node->GetOpDesc(), "op_unique_key", "ifa_key");
  (void)ge::AttrUtils::SetInt(ifa_node->GetOpDesc(), "ori_op_para_size", 24);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(ifa_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::ArgDesc> arg_descs;
  size_t arg_id = 0;
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, arg_id++);
  for (size_t i = 0; i < 3UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, arg_id);
  }
  arg_id++;
  for (size_t i = 0; i < 3UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, arg_id);
  }
  arg_id++;
  for (size_t i = 0; i < 12UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, arg_id++);
  }
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 0);
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(ifa_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 24 24 24 24 24 24 24 0 0 4 0 0 0 0 0 24 24 0 0 24 0 1 2 208 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckDynamicOutputSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::BatchSingleGraph();
  auto batch_node = graph->FindNode("Batch");
  ge::AttrUtils::SetStr(batch_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(batch_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(batch_node->GetOpDesc(), "ori_op_para_size", 24);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(batch_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::ArgDesc> arg_descs;
  for (size_t i = 0; i < 4UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT, 0);
  }
  for (size_t i = 0; i < 2UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 0);
  }
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 1);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 2);
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(batch_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 24 24 24 24 24 24 24 24 0 1 2 112 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckInstanceFormatSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::BatchSingleGraph();
  auto batch_node = graph->FindNode("Batch");
  ge::AttrUtils::SetStr(batch_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(batch_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(batch_node->GetOpDesc(), "ori_op_para_size", 24);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(batch_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::ArgDesc> arg_descs;
  for (size_t i = 0; i < 4UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_INSTANCE, 0);
  }
  for (size_t i = 0; i < 2UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_INSTANCE, 0);
  }
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_INSTANCE, 1);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_INSTANCE, 2);
  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(batch_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 24 24 24 24 24 24 24 24 0 1 2 112 ");
}


TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckDynamicOutputDescSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::BatchSingleGraph();
  auto batch_node = graph->FindNode("Batch");
  ge::AttrUtils::SetStr(batch_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(batch_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(batch_node->GetOpDesc(), "ori_op_para_size", 24);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(batch_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::ArgDesc> arg_descs;
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_DESC, 0, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_DESC, 0, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 1);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 2);

  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(batch_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 208 112 24 24 0 1 2 80 ");
}

TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingMemCheckHiddenInputSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::BatchSingleGraph();
  auto batch_node = graph->FindNode("Batch");
  ge::AttrUtils::SetStr(batch_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  (void)ge::AttrUtils::SetBool(batch_node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(batch_node->GetOpDesc(), "ori_op_para_size", 24);
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(batch_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::ArgDesc> arg_descs;
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_DESC, 0, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::HIDDEN_INPUT, 1);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_DESC, 0, true);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 1);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT, 2);

  std::string memcheck_info;
  ret = TilingMemCheck::ConstructMemCheckInfo(batch_node->GetOpDesc(), run_info, arg_descs, memcheck_info);
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string memcheck_info_str = parse_int(memcheck_info);
  EXPECT_EQ(memcheck_info_str, "0 208 32 112 24 24 0 1 2 88 ");
}


TEST_F(RegisterOpTilingRT2UT, AicoreParseAndTilingDfxInstanceFormatSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::BatchSingleGraph();
  auto batch_node = graph->FindNode("Batch");
  ge::AttrUtils::SetStr(batch_node->GetOpDesc(), COMPILE_INFO_JSON, "testst");
  utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(batch_node);
  fe::PlatFormInfos platform_infos;
  graphStatus ret = AicoreRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::ArgDesc> arg_descs;
  // 解析时不占位
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::FFTS_ADDR, 0);

  for (size_t i = 0; i < 4UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::INPUT_INSTANCE, 0);
  }
  for (size_t i = 0; i < 2UL; i++) {
    ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_INSTANCE, 0);
  }
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_INSTANCE, 1);
  ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::OUTPUT_INSTANCE, 2);

  std::vector<int64_t> args_size_list;
  std::vector<ArgsIndexToIoIndex> args_index_to_io_index;

  ret = TilingDfx::GetArgsSizeWithArgsFormat(batch_node->GetOpDesc(), arg_descs, args_size_list, args_index_to_io_index);
  EXPECT_EQ(ret, ge::SUCCESS);

  std::vector<int64_t> args_size_list_expect(8, 0);
  EXPECT_EQ(args_size_list.size(), args_size_list_expect.size());
  for (size_t i = 0U; i < args_size_list.size(); i++) {
    EXPECT_EQ(args_size_list[i], args_size_list_expect[i]);
  }

  std::vector<ArgsIndexToIoIndex> args_index_to_io_index_expect =
  {{ArgsRole::kInput, 0, 0}, {ArgsRole::kInput, 1, 0}, {ArgsRole::kInput, 2, 0},
   {ArgsRole::kInput, 3, 0}, {ArgsRole::kOutput, 4, 0}, {ArgsRole::kOutput, 5, 0},
   {ArgsRole::kOutput, 6, 1}, {ArgsRole::kOutput, 7, 2}};

  EXPECT_EQ(args_index_to_io_index.size(), args_index_to_io_index_expect.size());
  for (size_t i = 0U; i < args_index_to_io_index.size(); i++) {
    EXPECT_EQ(args_index_to_io_index[i].args_role, args_index_to_io_index_expect[i].args_role);
    EXPECT_EQ(args_index_to_io_index[i].args_index, args_index_to_io_index_expect[i].args_index);
    EXPECT_EQ(args_index_to_io_index[i].io_index, args_index_to_io_index_expect[i].io_index);
  }
}

TEST_F(RegisterOpTilingRT2UT, AicoreWithAtomicParseAndTilingSuccess) {
  typedef void* (*CreateCompileInfo)();
  typedef void (*DeleteCompileInfo)(void *obj);
  CreateCompileInfo create_compile_info = []() -> void *{
    auto info = new DynamicAtomicAddrCleanCompileInfo();
    info->core_num = 8;
    info->ub_size = 131072;
    info->workspace_num = 1;
    info->_workspace_index_list.emplace_back(0);
    return info;
  };

  DeleteCompileInfo delete_compile_info = [](void *obj) -> void {
    if (obj != nullptr) {
      delete (DynamicAtomicAddrCleanCompileInfo *)obj;
      obj = nullptr;
    }
  };

  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  // mock dynamic atomic clean tiling func
  auto op_impl_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("MemSet");
  op_impl_func->tiling = DummyTilingForDynamicAtomicAddrClean;
  op_impl_func->tiling_parse = MemsetTilingParse;
  op_impl_func->compile_info_creator = create_compile_info;
  op_impl_func->compile_info_deleter = delete_compile_info;

  auto graph = ShareGraph::BuildAtomicAicoreGraph();
  auto trans1_node = graph->FindNode("trans1");
  trans1_node->GetOpDesc()->SetWorkspaceBytes({256,1,2}); // simulate trans1 node finished tiling
  (void)ge::AttrUtils::SetListInt(trans1_node->GetOpDesc(), "tbe_op_atomic_dtypes", {0});
  ge::TensorUtils::SetSize(*trans1_node->GetOpDesc()->MutableOutputDesc(0), 128);
  std::map<int64_t, int64_t> index_2_workspace_size = {{0,5}};
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info = {{"trans1", index_2_workspace_size}};
  trans1_node->GetOpDesc()->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);

  utils::OpRunInfo run_info;
  fe::PlatFormInfos platform_infos;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(trans1_node);
  graphStatus ret = AtomicRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // expect value
  std::vector<std::pair<std::string, int64_t>> expect_value = {{"core_num", 8},         {"ub_size", 131072},
                                                               {"workspace_num", 1},    {"workspace_index", 0},
                                                               {"clean_out_size", 128}, {"clean_workspace_size", 256}};
  auto workspace = run_info.GetAllWorkspaces();
  for (size_t i = 0; i < run_info.GetWorkspaceNum(); ++i) {
    EXPECT_EQ(workspace[i], expect_value[i].second);
  }
}

TEST_F(RegisterOpTilingRT2UT, AicoreWithMemsetParseAndTilingSuccess) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  // mock dynamic atomic clean tiling func
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("MemSet")->tiling
      = DummyTilingForDynamicAtomicAddrClean;

  auto graph = ShareGraph::BuildMemSetAicoreGraph();
  auto memset_node = graph->FindNode("memset");
  memset_node->GetOpDesc()->SetWorkspaceBytes({256, 32}); // simulate trans1 node finished tiling

  utils::OpRunInfo run_info;
  fe::PlatFormInfos platform_infos;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(memset_node);
  graphStatus ret = AtomicRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingRT2UT, FftsRtParseAndTilingSuccess) {
  auto graph = ShareGraph::ConcatV2ConstDependencyGraph();
  auto node = graph->FindNode("concatv2");
  const auto &op_desc = node->GetOpDesc();
  const Operator op = OpDescUtils::CreateOperatorFromNode(node);

  ffts::ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ffts::ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vec_1.push_back(1);
  vector<vector<int64_t>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  std::vector<uint32_t> input_tensor_indexes = {0,1};
  std::vector<uint32_t> output_tensor_indexes = {0};
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_indexes = input_tensor_indexes;
  slice_info_ptr->output_tensor_indexes = output_tensor_indexes;

  (void)op_desc->SetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info_ptr);
  GeShape shape({4,1,3,4,16});
  std::vector<OpRunInfoV2> op_run_infos;
  fe::PlatFormInfos platform_infos;
  // without compile info json, tiling return failed
  EXPECT_EQ(FftsRtParseAndTiling(op, platform_infos, op_run_infos), ge::GRAPH_FAILED);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  auto dstAnchor = node->GetInDataAnchor(0);
  ge::AnchorUtils::SetStatus(dstAnchor, ge::ANCHOR_DATA);
  EXPECT_EQ(FftsRtParseAndTiling(op, platform_infos, op_run_infos), ge::GRAPH_SUCCESS);
}

// slice instance over
TEST_F(RegisterOpTilingRT2UT, OpFftsPlusCalculate_2) {
  auto graph = ShareGraph::AicoreGraph();
  auto node = graph->FindNode("add1");
  const auto &op_desc = node->GetOpDesc();
  const Operator op = OpDescUtils::CreateOperatorFromNode(node);

  ffts::ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ffts::ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vec_1.push_back(1);
  vector<vector<int64_t>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 4;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  (void)op_desc->SetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info_ptr);
  GeShape shape({4,1,3,4,16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  std::vector<OpRunInfoV2> op_run_infos;
  fe::PlatFormInfos platform_infos;
  EXPECT_EQ(FftsRtParseAndTiling(op, platform_infos, op_run_infos), ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingRT2UT, SoftSyncOpRtParseAndTiling) {
  auto graph = ShareGraph::AicoreGraph();
  auto node = graph->FindNode("add1");
  const auto &op_desc = node->GetOpDesc();
  const Operator op = OpDescUtils::CreateOperatorFromNode(node);

  ffts::ThreadSliceMapDyPtr slice_info_ptr = std::make_shared<ffts::ThreadSliceMapDy>();
  vector<int64_t> vec_1;
  vec_1.push_back(1);
  vector<vector<int64_t>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 4;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_indexes.push_back(0);
  slice_info_ptr->output_tensor_indexes.push_back(0);
  (void)op_desc->SetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info_ptr);
  GeShape shape({4,1,3,4,16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  EXPECT_TRUE(ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key));
  EXPECT_TRUE(ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json));
  EXPECT_TRUE(ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true));
  fe::PlatFormInfos platform_infos;
  OpRunInfoV2 run_info;
  run_info.SetLocalMemorySize(10U);  // test: local memory size will be updated in SoftSyncOpRtParseAndTiling
  auto space_registry = SpaceRegistryFaker().Build();
  // TODOO：用例适配
  auto ret = SoftSyncOpRtParseAndTiling(op, platform_infos, run_info, space_registry);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(run_info.GetLocalMemorySize(), 0U);
}
} // namespace optiling

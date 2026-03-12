/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/aicore/converter/autofuse_node_converter.h"
#include "engine/aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "engine/node_converter_utils.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_builder/value_holder_generator.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/const_data_helper.h"
#include "graph/utils/graph_dump_utils.h"
#include <iostream>
#include "engine/aicore/kernel/autofuse_op_kernel.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "op_stub/autofuse_impl/autofuse_stub.h"
#include "faker/kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "stub/gert_runtime_stub.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/graph/utils/file_utils.h"

using namespace ge;
using namespace gert::bg;

namespace gert {
namespace kernel {
std::vector<std::string> InferShapeKernelTrace(const KernelContext *context);
ge::graphStatus BuildCacheableTilingFwkDataOutput(const ge::FastNode *node, KernelContext *context);
ge::graphStatus PrepareCacheableTilingFwkData(KernelContext *context);
ge::graphStatus CacheableTiling(KernelContext *context);
ge::graphStatus PrepareTilingFwkData(KernelContext *context);
ge::graphStatus BuildTilingFwkDataOutput(const ge::FastNode *node, KernelContext *context);
}

namespace {
using DfxInputSymbolInfo = ge::graphStatus(*)(const KernelContext *, char *, size_t);
std::string GetAutofuseSoPath() {
  std::string cmake_binary_path = CMAKE_BINARY_DIR;
  return cmake_binary_path + "/tests/depends/op_stub/libautofuse_stub.so";
}

ge::ComputeGraphPtr BuildAutofuseGraph() {
  auto graph = ShareGraph::AutoFuseNodeGraph();
  (void)ge::AttrUtils::SetInt(graph, "_all_symbol_num", 8);
  auto fused_graph_node = graph->FindNode("fused_graph");
  auto fused_graph_node1 = graph->FindNode("fused_graph1");

  auto autofuse_stub_so = GetAutofuseSoPath();
  std::cout << "bin path: " << autofuse_stub_so << std::endl;
  (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "bin_file_path", autofuse_stub_so);
  (void)ge::AttrUtils::SetStr(fused_graph_node1->GetOpDesc(), "bin_file_path", autofuse_stub_so);

  (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "_symbol_infer_shape_cache_key", "xxxxx");
  (void)ge::AttrUtils::SetStr(fused_graph_node1->GetOpDesc(), "_symbol_infer_shape_cache_key", "xxxxx");

  return graph;
}

using ComputeNodeDesc = RtKernelLaunchArgsEx::ComputeNodeDesc;

void CreateDefaultArgsInfo(ArgsInfosDesc::ArgsInfo *args_info, size_t input_num, size_t output_num) {
  auto node_io_num = input_num + output_num;
  for (size_t idx = 0U; idx < node_io_num; ++idx) {
    int32_t start_index = (idx < input_num) ? idx : (idx - input_num);
    auto arg_type = (idx < input_num) ? ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT
                                      : ArgsInfosDesc::ArgsInfo::ArgsInfoType::OUTPUT;
    if (idx == 1) {
      args_info[idx].Init(arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, -1, 0U);
    } else {
      args_info[idx].Init(arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, start_index, 1U);
    }
  }
}
std::unique_ptr<uint8_t[]> CreateDefaultArgsInfoDesc(size_t input_num, size_t output_num) {
  const size_t args_info_num = input_num + output_num;
  ArgsInfosDesc::ArgsInfo args_info[args_info_num];
  CreateDefaultArgsInfo(args_info, input_num, output_num);
  size_t total_size = 0U;
  const size_t args_info_size = args_info_num * sizeof(ArgsInfosDesc::ArgsInfo);
  auto args_info_desc_holder = ArgsInfosDesc::Create(args_info_size, total_size);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  if (args_info_size > 0) {
    GELOGD("Copy args info to compute node extended desc mem, size:%lld", args_info_size);
    GE_ASSERT_EOK(memcpy_s(args_info_desc->MutableArgsInfoBase(), args_info_desc->GetArgsInfoSize(), args_info,
                           args_info_size) != EOK);
  }
  args_info_desc->Init(input_num, output_num, input_num, output_num);
  return args_info_desc_holder;
}

void *DlopenAutofuseSo() {
  auto so_path = GetAutofuseSoPath();
  char real_path[MMPA_MAX_PATH] = {};
  (void)mmRealPath(so_path.c_str(), &real_path[0], MMPA_MAX_PATH);
  return mmDlopen(real_path, static_cast<int32_t>(MMPA_RTLD_NOW));
}

void DlcloseSo(void *autofuse_so_handle) {
  if (autofuse_so_handle != nullptr) {
    (void)mmDlclose(autofuse_so_handle);
  }
}

void *DlopenSym(const std::string &func_name, void *autofuse_so_handle) {
  if (autofuse_so_handle == nullptr) {
    return nullptr;
  }
  return mmDlsym(autofuse_so_handle, func_name.c_str());
}

void InferTraceWithSymboInfoTest(const size_t all_sym_num, const std::string &expect_info) {
  auto compute_graph = BuildAutofuseGraph();
  auto op_desc = compute_graph->FindNode("fused_graph")->GetOpDesc();
  auto autofuse_so_handle = DlopenAutofuseSo();
  ASSERT_NE(autofuse_so_handle, nullptr);

  auto dfx_info_func = DlopenSym("DfxInputSymbolInfo", autofuse_so_handle);
  ASSERT_NE(dfx_info_func, nullptr);

  auto infer_shape_func = DlopenSym("InferShape", autofuse_so_handle);
  ASSERT_NE(infer_shape_func, nullptr);

  size_t input_num = 2;
  gert::Shape shape0 = gert::Shape({1, 2});
  gert::Shape shape1 = gert::Shape({1, 2});

  auto context_holder = gert::KernelRunContextBuilder()
      .Inputs({(void *)input_num, &shape0, &shape1, (void *)all_sym_num, dfx_info_func, infer_shape_func})
      .Build(op_desc);
  auto context = context_holder.GetKernelContext();
  std::vector<std::string> trace_info = gert::kernel::InferShapeKernelTrace(context);
  ASSERT_EQ(trace_info[1], expect_info);
}
}

class AutofuseNodeUT : public bg::BgTestAutoCreate3StageFrame {
 protected:
  void SetUp() override {
    //BgTest::SetUp();
    BgTestAutoCreate3StageFrame::SetUp();
  }
  void TearDown() override {
    BgTestAutoCreate3StageFrame::TearDown();
  }
};

TEST_F(AutofuseNodeUT, autofuse_convert_test) {
  auto graph = BuildAutofuseGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto fused_graph_node = graph->FindNode("fused_graph");
  auto compile_result = global_data.FindCompiledResult(fused_graph_node);
  ASSERT_NE(compile_result, nullptr);

  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  auto data3_ret = LoweringDataNode(graph->FindNode("data3"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  ASSERT_TRUE(data3_ret.result.IsSuccess());

  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0],
                              data2_ret.out_shapes[0], data3_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0],
                              data2_ret.out_addrs[0], data3_ret.out_addrs[0]},
                          &global_data};

  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  auto autofuse_ret = LoweringAutofuseNode(fused_graph_node, add_input);
  ASSERT_TRUE(autofuse_ret.result.IsSuccess());
  ASSERT_EQ(autofuse_ret.out_addrs.size(), 1);
  ASSERT_EQ(autofuse_ret.out_shapes.size(), 1);
  ASSERT_EQ(autofuse_ret.order_holders.size(), 1);

  // cacheable tiling cache
  auto fused_graph_node1 = graph->FindNode("fused_graph1");
  autofuse_ret = LoweringAutofuseNode(fused_graph_node1, add_input);
  ASSERT_TRUE(autofuse_ret.result.IsSuccess());
  ASSERT_EQ(autofuse_ret.out_addrs.size(), 1);
  ASSERT_EQ(autofuse_ret.out_shapes.size(), 1);
  ASSERT_EQ(autofuse_ret.order_holders.size(), 1);

  auto exe_graph = autofuse_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(autofuse_ret.out_addrs), autofuse_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);

  size_t sym_tiling_node = 0;
  size_t cacheable_sym_tiling_node = 0;
  size_t pre_tiling_cache_node = 0;
  size_t get_tiling_cache_key_node = 0;
  size_t infer_shape_node = 0;
  size_t tiling_parse_node = 0;
  for (auto node : execute_graph->GetAllNodes()) {
    auto node_type = node->GetType();
    if (node_type == "Tiling") {
      sym_tiling_node++;
    } else if (node_type == "CacheableTiling") {
      cacheable_sym_tiling_node++;
    } else if (node_type == "PrepareCacheableTilingFwkData") {
      pre_tiling_cache_node++;
    } else if (node_type == "GetSymbolTilingCacheKey") {
      get_tiling_cache_key_node++;
    } else if (node_type == "InferShape") {
      infer_shape_node++;
    } else if (node_type == "SymbolTilingParse") {
      tiling_parse_node++;
    }
  }
  // 两个asc节点一个开起来缓存，两个节点infershapekey相同可以merge
  EXPECT_EQ(sym_tiling_node, 0);
  EXPECT_EQ(cacheable_sym_tiling_node, 2);
  EXPECT_EQ(pre_tiling_cache_node, 2);
  EXPECT_EQ(get_tiling_cache_key_node, 2);
  EXPECT_EQ(infer_shape_node, 1);
  EXPECT_EQ(tiling_parse_node, 2);
  DumpGraph(execute_graph.get(), "GeneralAutofuseExe");
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}

TEST_F(AutofuseNodeUT, symtiling_kernel_test) {
  auto compute_graph = BuildAutofuseGraph();
  auto op_desc = compute_graph->FindNode("fused_graph")->GetOpDesc();
  auto autofuse_so_handle = DlopenAutofuseSo();
  ASSERT_NE(autofuse_so_handle, nullptr);

  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
      .output_num = 1,
      .workspace_cap = 8,
      .max_tiling_data = 128,
      .need_shape_buffer = false,
      .need_overflow = false,
      .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  args->UpdateBaseByTilingSize(100);
  args->GetTilingData().SetDataSize(100);
  args->UpdateBaseArgsSize();

  auto tiling_func = DlopenSym("TilingFunc", autofuse_so_handle);

  auto fwk_data_holder = gert::KernelRunContextBuilder()
      .Inputs({tiling_func, args})
      .Outputs({nullptr})
      .Build(op_desc);
  auto fwk_data_context = fwk_data_holder.GetKernelContext();
  ASSERT_EQ(kernel::BuildTilingFwkDataOutput(nullptr, fwk_data_context), GRAPH_SUCCESS);
  ASSERT_EQ(kernel::PrepareTilingFwkData(fwk_data_context), GRAPH_SUCCESS);
  auto fwk_data = fwk_data_context->GetOutputPointer<kernel::TilingFwkData>(0U);
  ASSERT_NE(fwk_data, nullptr);

  gert::Shape shape0 = gert::Shape({1, 2}) ;
  gert::Shape shape1 = gert::Shape({1, 2});
  size_t input_data_num = 2;
  AfTilingParseData parse_data{8, 16 * 1024};

  const auto workspace_size_t = gert::ContinuousVector::Create<size_t>(16);
  auto autofuse_tiling_context_holder = gert::KernelRunContextBuilder()
      .Inputs({(void *)input_data_num, &shape0, &shape1, &parse_data, fwk_data, nullptr, nullptr})
      .Outputs({nullptr, nullptr, nullptr, nullptr,
                static_cast<void *>(workspace_size_t.get()), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
      .Build(op_desc);

  auto autofuse_tiling_context = autofuse_tiling_context_holder.GetKernelContext();
  ASSERT_EQ(kernel::Tiling(autofuse_tiling_context), GRAPH_SUCCESS);
  auto tiling_context = reinterpret_cast<TilingContext *>(autofuse_tiling_context);

  auto workspace_size = *(tiling_context->GetWorkspaceSizes(1));
  ASSERT_EQ(workspace_size, 1024);
  auto block_dim =
      *(autofuse_tiling_context->GetOutputPointer<uint32_t>(static_cast<size_t>(TilingContext::kOutputBlockDim)));
  ASSERT_EQ(block_dim, 8);
  auto tiling_data =
      autofuse_tiling_context->GetOutput(static_cast<size_t>(TilingContext::kOutputTilingData))->GetValue<void *>();
  ASSERT_NE(tiling_data, nullptr);

  free(node_desc_holder);
  DlcloseSo(autofuse_so_handle);
}

TEST_F(AutofuseNodeUT, cacheablesymtiling_kernel_test) {
  auto compute_graph = BuildAutofuseGraph();
  auto op_desc = compute_graph->FindNode("fused_graph")->GetOpDesc();
  auto autofuse_so_handle = DlopenAutofuseSo();
  ASSERT_NE(autofuse_so_handle, nullptr);

  size_t compiled_args_size = 64;
  auto node_desc_holder = malloc(compiled_args_size + sizeof(ComputeNodeDesc));
  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(node_desc_holder);
  *node_desc = {.input_num = 1,
      .output_num = 1,
      .workspace_cap = 8,
      .max_tiling_data = 128,
      .need_shape_buffer = false,
      .need_overflow = false,
      .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  ASSERT_NE(holder, nullptr);
  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  args->UpdateBaseByTilingSize(100);
  args->GetTilingData().SetDataSize(100);
  args->UpdateBaseArgsSize();

  // PrepareSymbolTilingCache
  size_t all_symbol_num = 16;
  gert::Shape shape0 = gert::Shape({1, 2});
  gert::Shape shape1 = gert::Shape({1, 2});
  size_t input_num = 2;
  auto tiling_func = DlopenSym("TilingFunc", autofuse_so_handle);
  ASSERT_NE(tiling_func, nullptr);
  auto func_getsymtilingcache = DlopenSym("GetSymbolTilingCacheKey", autofuse_so_handle);
  ASSERT_NE(func_getsymtilingcache, nullptr);

  auto get_tiling_cahce_key_holder = gert::KernelRunContextBuilder()
      .Inputs({(void *)input_num, &shape0, &shape1, func_getsymtilingcache, (void *)all_symbol_num})
      .Outputs({nullptr})
      .Build(op_desc);
  auto get_tiling_cahce_context = get_tiling_cahce_key_holder.GetKernelContext();
  ASSERT_EQ(kernel::BuildSymbolTilingCacheKeyOutputs(nullptr, get_tiling_cahce_context),ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel::GetSymbolTilingCacheKeyKernel(get_tiling_cahce_context), ge::GRAPH_SUCCESS);
  auto all_sym_num_vector = get_tiling_cahce_context->GetOutputPointer<TypedContinuousVector<int64_t>>(0);
  ASSERT_NE(all_sym_num_vector, nullptr);
  ASSERT_EQ(all_sym_num_vector->GetCapacity(), 16U);

  const size_t data_dependency = 0U;
  std::string func_name = "BuildSymbolTilingCacheKey";
  auto cache_tiling_fwk_holder = gert::KernelRunContextBuilder()
      .Inputs({tiling_func, args, (void *)data_dependency, func_name.data()})
      .Outputs({nullptr})
      .Build(op_desc);
  auto cache_tiling_fwk_context = cache_tiling_fwk_holder.GetKernelContext();
  ASSERT_EQ(kernel::BuildCacheableTilingFwkDataOutput(nullptr, cache_tiling_fwk_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel::PrepareCacheableTilingFwkData(cache_tiling_fwk_context), ge::GRAPH_SUCCESS);
  auto tiling_fwk_data = cache_tiling_fwk_context->GetOutputPointer<kernel::CacheableTilingFwkData>(0UL);
  ASSERT_NE(tiling_fwk_data, nullptr);

  AfTilingParseData parse_data{8, 16 * 1024};
  const auto cacheable_workspace_size_t = gert::ContinuousVector::Create<size_t>(16);
  auto cacheable_tiling_context_holder = gert::KernelRunContextBuilder()
      .Inputs({(void *)input_num, &shape0, &shape1, &parse_data, all_sym_num_vector, tiling_fwk_data, nullptr, nullptr})
      .Outputs({nullptr, nullptr, nullptr, nullptr,
                static_cast<void *>(cacheable_workspace_size_t.get()), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
      .Build(op_desc);
  auto cacheable_tiling_cxt = cacheable_tiling_context_holder.GetKernelContext();
  // add cache
  ASSERT_EQ(kernel::CacheableTiling(cacheable_tiling_cxt), GRAPH_SUCCESS);
  auto launch_arg = cacheable_tiling_cxt->GetOutputPointer<RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg));
  ASSERT_EQ(launch_arg->GetArgsCacheInfo().cache_status, RtKernelLaunchArgsEx::TilingCacheStatus::kMissed);
  // apply cache
  ASSERT_EQ(kernel::CacheableTiling(cacheable_tiling_cxt), GRAPH_SUCCESS);
  launch_arg = cacheable_tiling_cxt->GetOutputPointer<RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg));
  ASSERT_EQ(launch_arg->GetArgsCacheInfo().cache_status, RtKernelLaunchArgsEx::TilingCacheStatus::kHit);

  free(node_desc_holder);
  DlcloseSo(autofuse_so_handle);
}

TEST_F(AutofuseNodeUT, infer_trace_test) {
  auto context_holder = KernelRunContextFaker().NodeType("test1").Build();
  auto context = context_holder.GetContext<KernelContext>();

  std::vector<std::string> trace_info = gert::kernel::InferShapeKernelTrace(context);
  ASSERT_EQ(trace_info[1], "input original shapes : , input storage shapes : ");
}

TEST_F(AutofuseNodeUT, symbol_tiling_parse_kernel_test) {
  auto compute_graph = BuildAutofuseGraph();
  auto op_desc = compute_graph->FindNode("fused_graph")->GetOpDesc();
  auto autofuse_so_handle = DlopenAutofuseSo();
  ASSERT_NE(autofuse_so_handle, nullptr);

  auto func_tilingparse = DlopenSym("TilingParse", autofuse_so_handle);
  ASSERT_NE(func_tilingparse, nullptr);

  fe::PlatFormInfos platform_infos;
  auto tilingparse_context_holder = gert::KernelRunContextBuilder()
      .Inputs({&platform_infos, func_tilingparse})
      .Outputs({nullptr})
      .Build(op_desc);
  auto tilingparse_context = tilingparse_context_holder.GetKernelContext();
  ASSERT_EQ(kernel::SymbolTilingParseKernel(tilingparse_context), GRAPH_SUCCESS);
  auto tilingparse_data = tilingparse_context->GetOutputPointer<AfTilingParseData *>(0U);
  ASSERT_NE(tilingparse_data, nullptr);
  ASSERT_EQ((*tilingparse_data)->aiv_num, 8U);
  ASSERT_EQ((*tilingparse_data)->ub_size, 184 * 1024);
  DlcloseSo(autofuse_so_handle);
}

TEST_F(AutofuseNodeUT, dlopen_autofuseso_funcs_kernel_test) {
  auto compute_graph = BuildAutofuseGraph();
  auto op_desc = compute_graph->FindNode("fused_graph")->GetOpDesc();

  auto so_path = GetAutofuseSoPath();
  auto context_holder = gert::KernelRunContextBuilder()
      .Inputs({so_path.data()})
      .Outputs({nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
      .Build(op_desc);
  auto context = context_holder.GetKernelContext();
  ASSERT_EQ(kernel::GetAutofuseFuncsKernel(context), GRAPH_SUCCESS);
}

TEST_F(AutofuseNodeUT, infer_trace_symbol_info_test1) {
  InferTraceWithSymboInfoTest(16, "Symbolic infos: s0: 1, s1: 2, s2: 1.");
}

TEST_F(AutofuseNodeUT, infer_trace_symbol_info_test2) {
  InferTraceWithSymboInfoTest(1, "Symbolic infos: s0: 1, s1: 2");
}

TEST_F(AutofuseNodeUT, infer_trace_symbol_info_test4) {
  InferTraceWithSymboInfoTest(0, "Symbolic infos: no symbol.");
}

TEST_F(AutofuseNodeUT, autofuse_so_offline_test) {
  auto graph = BuildAutofuseGraph();
  auto fused_graph_node = graph->FindNode("fused_graph");
  auto op_desc = fused_graph_node->GetOpDesc();

  auto autofuse_stub_so = GetAutofuseSoPath();
  std::cout << "bin path: " << autofuse_stub_so << std::endl;

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  LowerInput data_input = {{}, {}, &global_data};
  auto compile_result = global_data.FindCompiledResult(fused_graph_node);
  ASSERT_NE(compile_result, nullptr);

  // 读取so并转换成二进制
  uint32_t bin_len = 0U;
  auto bin = ge::GetBinDataFromFile(autofuse_stub_so, bin_len);
  const auto &pos = autofuse_stub_so.find_last_of("/");
  ASSERT_TRUE(pos != std::string::npos);
  const auto &so_name = autofuse_stub_so.substr(pos + 1UL);
  const auto &vendor_name = autofuse_stub_so.substr(0, pos);
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new(std::nothrow) char[bin_len]);
  std::string so_bin_str(bin.get(), bin_len);
  (void) memcpy_s(so_bin.get(), bin_len, so_bin_str.c_str(), bin_len);
  ge::OpSoBinPtr so_bin_ptr = ge::MakeShared<ge::OpSoBin>(so_name, vendor_name, std::move(so_bin), bin_len);

  // 修改路径为新创建的so
  std::string so_path_for_test = vendor_name + "/1.so";
  (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "bin_file_path", so_path_for_test);
  system(("rm -f " + so_path_for_test).c_str());

  std::map<std::string, ge::OpSoBinPtr> bin_file_buffer_map;
  bin_file_buffer_map[so_path_for_test] = so_bin_ptr;
  // 创建bin_file_buffer
  graph->SetExtAttr<std::map<string, OpSoBinPtr>>("bin_file_buffer", bin_file_buffer_map);

  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  auto data3_ret = LoweringDataNode(graph->FindNode("data3"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  ASSERT_TRUE(data3_ret.result.IsSuccess());

  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0],
                              data2_ret.out_shapes[0], data3_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0],
                              data2_ret.out_addrs[0], data3_ret.out_addrs[0]},
                          &global_data};

  auto autofuse_ret = LoweringAutofuseNode(fused_graph_node, add_input);
  ASSERT_TRUE(autofuse_ret.result.IsSuccess());

  auto context_holder = gert::KernelRunContextBuilder()
      .Inputs({so_path_for_test.data()})
      .Outputs({nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
      .Build(op_desc);
  auto context = context_holder.GetKernelContext();
  ASSERT_EQ(kernel::GetAutofuseFuncsKernel(context), GRAPH_SUCCESS);

  (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "bin_file_path", autofuse_stub_so);
  graph->DelExtAttr("bin_file_buffer");
}

TEST_F(AutofuseNodeUT, autofuse_so_offline_no_bin_file_path_test) {
  auto graph = ShareGraph::AutoFuseNodeGraph();
  (void)ge::AttrUtils::SetInt(graph, "_all_symbol_num", 8);
  auto fused_graph_node = graph->FindNode("fused_graph");
  auto op_desc = fused_graph_node->GetOpDesc();

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  LowerInput data_input = {{}, {}, &global_data};
  auto compile_result = global_data.FindCompiledResult(fused_graph_node);
  ASSERT_NE(compile_result, nullptr);

  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  auto data3_ret = LoweringDataNode(graph->FindNode("data3"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  ASSERT_TRUE(data3_ret.result.IsSuccess());

  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0],
                              data2_ret.out_shapes[0], data3_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0],
                              data2_ret.out_addrs[0], data3_ret.out_addrs[0]},
                          &global_data};

  auto autofuse_ret = LoweringAutofuseNode(fused_graph_node, add_input);
  ASSERT_FALSE(autofuse_ret.result.IsSuccess());

  graph->DelExtAttr("bin_file_buffer");
}

TEST_F(AutofuseNodeUT, autofuse_so_offline_no_bin_file_buffer_test) {
  auto graph = BuildAutofuseGraph();
  auto fused_graph_node = graph->FindNode("fused_graph");
  auto op_desc = fused_graph_node->GetOpDesc();

  auto autofuse_stub_so = GetAutofuseSoPath();
  std::cout << "bin path: " << autofuse_stub_so << std::endl;

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  LowerInput data_input = {{}, {}, &global_data};
  auto compile_result = global_data.FindCompiledResult(fused_graph_node);
  ASSERT_NE(compile_result, nullptr);

  // 读取so并转换成二进制
  uint32_t bin_len = 0U;
  auto bin = ge::GetBinDataFromFile(autofuse_stub_so, bin_len);
  const auto &pos = autofuse_stub_so.find_last_of("/");
  ASSERT_TRUE(pos != std::string::npos);
  const auto &so_name = autofuse_stub_so.substr(pos + 1UL);
  const auto &vendor_name = autofuse_stub_so.substr(0, pos);
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new(std::nothrow) char[bin_len]);
  std::string so_bin_str(bin.get(), bin_len);
  (void) memcpy_s(so_bin.get(), bin_len, so_bin_str.c_str(), bin_len);
  ge::OpSoBinPtr so_bin_ptr = ge::MakeShared<ge::OpSoBin>(so_name, vendor_name, std::move(so_bin), bin_len);

  // 修改路径为新创建的so
  std::string so_path_for_test = vendor_name + "/1.so";
  system(("rm -f " + so_path_for_test).c_str());

  std::map<std::string, ge::OpSoBinPtr> bin_file_buffer_map;
  bin_file_buffer_map[so_path_for_test] = so_bin_ptr;
  // 创建bin_file_buffer
  graph->SetExtAttr<std::map<string, OpSoBinPtr>>("bin_file_buffer", bin_file_buffer_map);

  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  auto data3_ret = LoweringDataNode(graph->FindNode("data3"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  ASSERT_TRUE(data3_ret.result.IsSuccess());

  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0],
                              data2_ret.out_shapes[0], data3_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0],
                              data2_ret.out_addrs[0], data3_ret.out_addrs[0]},
                          &global_data};

  auto autofuse_ret = LoweringAutofuseNode(fused_graph_node, add_input);
  ASSERT_FALSE(autofuse_ret.result.IsSuccess());

  graph->DelExtAttr("bin_file_buffer");
}
}
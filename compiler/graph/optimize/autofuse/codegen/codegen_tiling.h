/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_TILING_H__
#define __CODEGEN_TILING_H__
#include <sstream>
#include "ascir.h"
#include "schedule_result.h"

namespace codegen {
const std::string kTilingHeadIdentify = "TilingHead";
const std::string kTilingDataIdentify = "TilingData";
const std::string kTilingHeadGuard = "__AUTOFUSE_TILING_FUNC_COMMON_H__";
const std::string kTilingHeadInclude = "#include \"autofuse_tiling_func_common.h\"";
const std::string kTilingHeadCceKtTestGuard = "#ifndef __CCE_KT_TEST__";
const std::string kTilingHeadEndGuard = "#endif";
const std::string kTilingHeadTilingContext = "#include \"exe_graph/runtime/tiling_context.h\"";
const std::string kTilingDefAndConstIdentify = "tiling_def_and_tiling_const";
const std::string kCubeTilingHeadInclude = "#include \"autofuse_cube_tiling_data.h\"";

  using TilingLibCodegenFunc = bool (*)(const std::string &op_name,
                                        const ::ascir::FusedScheduledResult& fused_schedule_result,
                                        std::map<std::string, std::string> &options,
                                        std::map<std::string, std::string> &tiling_file_name_to_content, bool is_inductor_scene);
  struct PgoShapeStringStream {
    std::stringstream shape_dim_def;
    std::stringstream tiling_set_shape_dim;
    std::stringstream shape_dim_use;
  };
  class TilingLib {
   public:
    TilingLib(const std::string &lib_path, const std::string &codegen_symbol_name);
    std::map<std::string, std::string> Generate(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                const std::map<std::string, std::string> &shape_info,
                                                const std::string& pgo_dir,
                                                const std::string &core_num) const;
    std::map<std::string, std::string> GenerateForInductor(
        const ::ascir::FusedScheduledResult &fused_schedule_result) const;

    std::string GenerateForPgo(const ::ascir::FusedScheduledResult &fused_schedule_result, const std::string& pgo_dir) const;
    std::string GetTilingIncludeHead(bool is_cv = false) const;
   protected:
    std::string TilingFuncDef(const ::ascir::FusedScheduledResult& fused_schedule_result, 
                              const std::map<std::string, std::string> &shape_info, const std::string& pgo_dir,
                              const std::string &core_num) const;
    std::string TilingFuncDefForInductor(const ::ascir::FusedScheduledResult& fused_schedule_result) const;
    std::map<std::string, std::string> GetTilingHeaders(const ::ascir::FusedScheduledResult& fused_schedule_result, 
                                 bool is_inductor_scene, bool is_cv = false) const;
    std::string InferShapeDef(const ::ascir::HintGraph &graph) const;
    std::string OpDef(const ::ascir::HintGraph &graph) const;

    std::string OpInputDef(const ::ascir::NodeView& node) const;
    std::string OpOutputDef(const ::ascir::NodeView& node) const;

    std::string ExternFunctionDeclare(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                      const std::string tiling) const;
    std::string PGOProfilingCallbackDef(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                        const std::string tiling) const;
    std::string PGOSearchFuncInputOutputCallBackDef(const ::ascir::FusedScheduledResult& fused_schedule_result) const;
    std::string PGOSearchFuncInputOutputDef(const ::ascir::FusedScheduledResult& fused_schedule_result) const;
    std::string PGOSearchFuncInputOutputCall(const ::ascir::FusedScheduledResult& fused_schedule_result) const;
    std::string PGOSearchStructInputOutputDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string PGOSearchTensorInputOutputDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string PGOSearchFuncInputOutputStructAssignDef(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                        const std::string &struct_var_name) const;
    uint32_t PGOSearchFuncGetInputOutputCount(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string CalculateTensorMemorySizeStr(const ::ascir::TensorAttr& tensor) const;
    std::string PGOSearchTensorMallocDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string PGOSearchTensorFreeDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string StubHeadersWithoutCodegenFunc() const;
    std::string GetStubTilingHeaders(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string GenGetAutoFuseTilingInput(bool is_inductor_scene) const;
    std::string GenGetResLimitStru(void) const;
    bool IsMixKernelTaskType(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
   private:
    // 判断某个 origin_var 是否被特定 schedule_group 使用
    bool IsVarUsedInScheduleGroup(const std::string &var_define,
                                  const ::ascir::ScheduleGroup &schedule_group) const;
    std::string GenGetTilingSizeFunc(const std::string graph_name, const std::string tiling) const;
    std::string GenTilingFunc(const std::map<std::string, std::string> &shape_info,
                              const ::ascir::FusedScheduledResult& fused_schedule_result, const std::string func,
                      const std::string tiling, const std::string &core_num) const;
    std::string GenTilingFuncForInductor(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                         const std::string func, const std::string tiling) const;
    std::string GenPgoTilingFunc(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                 const std::string& tiling,
                                 codegen::PgoShapeStringStream &pgo_shape_dim,
                        bool is_inductor_scene, const std::string &core_num = "0") const;
    std::string GenPgoAutofuseTiling(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                     codegen::PgoShapeStringStream &pgo_shape_dim,
                                     const std::string &tiling, bool is_inductor_scene) const;

    std::string GenPgoTilingSearchPGO(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                      codegen::PgoShapeStringStream &pgo_shape_dim, 
                                      const std::string &tiling, bool is_inductor_scene, const std::string &core_num) const;

    std::string GenPgoTilingSearch(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                   codegen::PgoShapeStringStream &pgo_shape_dim,
                                   const std::string &tiling) const;
    std::string GenProfilingAllTilingData(std::string tiling_data_list_name,
                                          std::string tiling_data_perf_list_name,
                                          const ::ascir::FusedScheduledResult& fused_schedule_result,
                                          bool is_inductor_scene) const;
    std::string GenGetMaxBlockDimFromInput(const std::string &core_num) const;
    std::string GenPgoTilingSearchByCoreNum(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                            codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
  					                        bool is_inductor_scene, const std::string &core_num) const;
    std::string GenPGOGetTilingKey(const std::string tiling) const;
    std::string GenSavePGOSearchTilingDataFunc(const std::string tiling) const;
    std::string GenSavePGOConfigTilingDataFunc() const;
    void GenPgoSaveTilingKey(std::stringstream& ss) const;
    void GenPgoAppendSearchTilingData(std::stringstream& ss) const;
    void GenPgoKernelLaunchOpArgs(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenDynamicLibraryLoaderCode(std::stringstream &ss) const;
    void GenPgoHeaders(std::stringstream &ss) const;
    void GenPgoMain(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoEnvInit(const ::ascir::FusedScheduledResult &fused_schedule_result,
                       std::stringstream &ss) const;
    void GenPgoCardLock(std::stringstream &ss) const;
    void GenPgoMixTilingTable(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoCheckTilingIsMix(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoToolFunction(const ::ascir::FusedScheduledResult &fused_schedule_result, const std::string &pgo_dir,
                std::stringstream &ss) const;
    void GenPgoLaunchKernelInit(std::stringstream &ss) const;
    void GenPgoLaunchParamsInit(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoLaunchParamsDeInit(std::stringstream &ss) const;
    void GenPgoUpdateLaunchParams(std::stringstream &ss) const;
    void GenPgoLaunchParams(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoDeinit(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoWrapperParmCall(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoWrapperKernelLaunch(std::stringstream &ss) const;
    void GenPgoWrapper(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoProfilingConstants(std::stringstream &ss) const;
    void GenPgoMsptiStringTable(std::stringstream &ss) const;
    void GenPgoMsptiRequest(std::stringstream &ss) const;
    void GenPgoMsptiComplete(std::stringstream &ss) const;
    void GenPgoMsptiToolFunction(std::stringstream &ss) const;
    void GenPgoMsptiProfiling(std::stringstream &ss) const;
    void GenPgoBatchCallback(std::stringstream &ss) const;
    void GenPgoBatchProcess(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoGetProfilingBatch(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoProfilingCallback(std::stringstream &ss) const;
    void GenPgoGetProfiling(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoFunc(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoStaticFunc(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    void GenPgoProfiling(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
    std::string GenExternTilingFunc(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                    const std::map<std::string, std::string> &shape_info,
                                    const std::string tiling,
                                    const std::string &pgo_dir,
                                    const std::string &core_num) const;
    void TilingSetShapeDim(std::stringstream &tiling_set_shape_dim, const std::string &var_define,
                           const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string GenTilingCacheFunc(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                   const std::map<std::string, std::string> &shape_info) const;
    void TilingMappingSymbolToTiling(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                     std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const;
    void TilingProcessSymbolToTiling(const ::ascir::ImplGraph &graph, size_t graph_num, size_t res_num, size_t group_num,
                                     std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const;
    std::string GenCheckStaticShapeFunc(bool is_static) const;
    std::string GenGetWorkspaceSizeFunc(const std::string &tiling, const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string GenImplGraphWorkspaceSize(const ::ascir::ImplGraph &graph, const std::string &tiling_data, uint32_t index) const;
    std::string GenDfxInputSymbolInfo(const ::ascir::FusedScheduledResult& fused_schedule_result,
                                      const std::map<std::string, std::string> &shape_info) const;
    std::string GenFindBestTilingKeyFunc(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                         const std::string &tiling_data_name) const;
    std::string GenGetTilingKeyCount(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string GenGetTilingKeyForStatic() const;
    std::string GenGetTilingKeyKernelTypeForStatic(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    std::string GenCVTilingFunc() const;
    std::string GenTilingDataBlockDimAndWss() const;
    std::map<std::string, std::string> GenerateCVFusion(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                        const std::map<std::string, std::string> &shape_info,
                                                        const std::string &pgo_dir, const std::string &core_num) const;
    std::string GenExternTilingFuncBody(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                        const std::map<std::string, std::string> &shape_info, const std::string &tiling,
                                        const std::string &pgo_dir) const;
    std::string GenAscirTilingAndLaunchFunc(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
    TilingLibCodegenFunc codegen_func_{nullptr};
    bool enable_autofuse_pgo_{false};
  };
  }  // namespace codegen

#endif

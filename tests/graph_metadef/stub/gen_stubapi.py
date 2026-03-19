#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import re
import sys
import logging

"""
    generate stub func body by return type
"""
RETURN_STATEMENTS = {
    'int32_t': '    return 0;',
    'int': '    return 0;',
    'int&': '    return __LINE__',
    'uint8_t': '    return 0;',
    'uint32_t': '    return 0U;',
    'int64_t': '    return 0;',
    'uint64_t': '    return 0U;',
    'size_t': '    return 0U;',
    'std::size_t': '    return 0U;',
    'float': '    return 0.0f;',
    'bool': '    return false;',
    'void*': '    return nullptr;',
    'char*': '    return nullptr;',
    'char_t*': '    return nullptr;',
    'ge::char_t*': '    return nullptr;',
    'std::string': '    return "";',
    'std::string&': '    static std::string s;\n    return s;',
    'DataType': '    return DT_FLOAT;',
    'ge::DataType': '    return ge::DT_FLOAT;',
    'graphStatus': '    return ge::GRAPH_SUCCESS;',
    'ge::graphStatus': '    return ge::GRAPH_SUCCESS;',
    'ge::Status': '    return ge::SUCCESS;',
    'Status': '    return 0U;',
    'Format': '    return Format();',
    'ge::Format': '    return ge::Format();',
    'Graph': '    return Graph();',
    'Graph&': '    return *this;',
    'TensorDesc': '    return TensorDesc();',
    'TensorDesc&': '    return *this;',
    'Tensor': '    return Tensor();',
    'Operator&': '    return *this;',
    'GNode': '    return GNode();',
    'GNodePtr': '    return nullptr;',
    'ConstGraphPtr': '    return nullptr;',
    'GraphPtr': '    return nullptr;',
    'Placement': '    return static_cast<Placement>(0);',
    'Ptr': '    return nullptr;',
    'InitValueNum': '',
    'Option': '    return IGNORE;',
    'InferenceContextPtr': '    return nullptr;',
    'SubgraphBuilder': '    return nullptr;',
    'OperatorImplPtr': '    return nullptr;',
    'OutHandler': '    return nullptr;',
    'std::vector<int64_t>': '    return {};',
    'std::vector<std::string>': '    return {};',
    'std::vector<std::string>&': '    static std::vector<std::string> vec;\n    return vec;',
    'std::vector<OpParamDef>': '    return {};',
    'std::vector<ge::DataType>&': '    return this->impl_->types;',
    'std::vector<ge::Format>&': '    return this->impl_->formats;',
    'ge::AscendString': '    return "";',
    'ge::AscendString& OpParamDef::': '    return this->impl_->name;',
    'ge::AscendString& OpAttrDef::': '    return this->impl_->name;',
    'ge::AscendString& OpAICoreConfig::': '    return this->impl_->cfg_info[key];',
    'ge::AscendString& OpDef::': '    return this->impl_->op_type;',
    'std::vector<ge::AscendString>& OpMC2Def::': '    return this->impl_->group_list;',
    'std::vector<ge::AscendString>& OpAICoreConfig::': '    return this->impl_->cfg_keys;',
    'std::vector<ge::AscendString>& OpDefFactory::': '    static std::vector<ge::AscendString> g_ops_list;\n'
                                                     '    return g_ops_list;',
    'InitValueType&': '    return this->impl_->init_value_type;',
    'InitValueNum&': '    return this->impl_->init_value;',
    'std::vector<ScalarVar>& OpParamDef::': '    return this->impl_->init_value_list;',
    'OpAttrDef& OpAttrDef::': '    return *this;',
    'OpDef& OpDef::': '    return *this;',
    'OpAICPUDef& OpAICPUDef::': '    return *this;',
    'std::vector<ge::AscendString>& OpAICPUDef::': '    return this->impl_->cfg_keys;',
    'std::map<ge::AscendString, ge::AscendString>& OpAICPUDef::': '    return this->impl_->cfg_info;',
    'ge::AscendString& OpAICPUDef::': '    return this->impl_->cfg_info[key];',
    'OpHostCPUDef& OpHostCPUDef::': '    return *this;',
    'std::vector<ge::AscendString>& OpHostCPUDef::': '    return this->impl_->cfg_keys;',
    'std::map<ge::AscendString, ge::AscendString>& OpHostCPUDef::': '    return this->impl_->cfg_info;',
    'ge::AscendString& OpHostCPUDef::': '    return this->impl_->cfg_info[key];',
    'OpMC2Def& OpMC2Def::': '    return *this;',
    'OpParamDef& OpParamDef::': '    return *this;',
    'OpMC2Def& OpDef::': '    return this->impl_->op_mc2;',
    'OpAICoreConfig& OpAICoreConfig::': '    return *this;',
    'OptionRegister& OptionRegister::': '    return *this;',
    'PassOptionRegister& PassOptionRegister::': '    return *this;',
    'std::map<ge::AscendString, ge::AscendString>& OpAICoreConfig::': '    return this->impl_->cfg_info;',
    'std::map<ge::AscendString, OpAICoreConfig>& OpAICoreDef::': '   return this->impl_->aicore_configs;',
    'OpAICoreDef& OpAICoreDef::': '    return *this;',
    'OpAICoreDef& OpDef::': '    return this->impl_->op_aicore;',
    'OpAICPUDef& OpDef::': '    return this->impl_->op_aicpu;',
    'OpHostCPUDef& OpDef::': '    return this->impl_->op_hostcpu;',
    'gert::OpImplKernelRegistry::TilingKernelFunc&': '    return this->impl_->tiling_func;',
    'OpParamDef& OpAICoreConfig::': '    static OpParamDef def("");\n    return def;',
    'std::vector<OpParamDef>& OpAICoreConfig::': '    static std::vector<OpParamDef> vec;\n    return vec;',
    'std::vector<OpParamDef>& OpDef::': '    static std::vector<OpParamDef> vec;\n    return vec;',
    'optiling::OP_CHECK_FUNC&': '    return this->impl_->op_chk_support;',
    'std::vector<std::vector<OpParamDef>>': '    return {};',
    'OpAttrDef& OpDef::': '    return this->impl_->attrs.back();',
    'optiling::PARAM_GENERALIZE_FUNC&': '    return this->impl_->op_generlize_func;',
    'OpParamDef& OpDef::': '    static OpParamDef def("");\n    return def;',
    'gert::OpImplKernelRegistry::InferShapeRangeKernelFunc&': '    return this->impl_->infer_shape_range;',
    'gert::OpImplKernelRegistry::InferShapeKernelFunc&': '    return this->impl_->infer_shape;',
    'gert::OpImplKernelRegistry::InferDataTypeKernelFunc&': '    return this->impl_->infer_data_type;',
    'std::vector<OpAttrDef>& OpDef::': '    return this->impl_->attrs;',
    'OpDef': '    return OpDef("default");',
    'ItemFindStatus': '    return ItemFindStatus::ITEM_FIND;',
    'OpImplRegisterV2& OpImplRegisterV2::': '    return *this;',
    'OpRunInfo& OpRunInfo::': '    return *this;',
    'uint8_t*': '    return nullptr;',
    'OpCompileInfo& OpCompileInfo::': '    return *this;',
    'OpImplKernelRegistry::OpImplFunctions*': '    return nullptr;',
    'OpImplRegistry& OpImplRegistry::': '    static OpImplRegistry instance;\n    return instance;',
    'DefaultOpImplSpaceRegistry& DefaultOpImplSpaceRegistry::':
        '    static DefaultOpImplSpaceRegistry instance;\n    return instance;',
    'StreamMngFuncRegistry& StreamMngFuncRegistry::':
        '    static StreamMngFuncRegistry instance;\n    return instance;',
    'OpImplRegistryHolderManager&': '    static OpImplRegistryHolderManager instance;\n    return instance;',
    'HiddenInputFuncRegistry&': '    static HiddenInputFuncRegistry instance;\n    return instance;',
    'CTilingDataClassFactory& CTilingDataClassFactory::GetInstance':
        '    static CTilingDataClassFactory instance;\n    return instance;',
    'FrameworkRegistry& FrameworkRegistry::': '    static FrameworkRegistry instance;\n    return instance;',
    'ScreenPrinter& ScreenPrinter::': '    static ScreenPrinter instance;\n    return instance;',
    'ErrorTracking& ErrorTracking::': '    static ErrorTracking instance;\n    return instance;',
    'HostCpuEngine& HostCpuEngine::': '    static HostCpuEngine instance;\n    return instance;',
    'OppSoManager& OppSoManager::': '    static OppSoManager instance;\n    return instance;',
    'OptionRegistry& OptionRegistry::': '    static OptionRegistry instance;\n    return instance;',
    'PassOptionRegistry& PassOptionRegistry::': '    static PassOptionRegistry instance;\n    return instance;',
    'RuntimePluginLoader& RuntimePluginLoader::': '    static RuntimePluginLoader instance;\n    return instance;',
    'OpImplRegister& OpImplRegister': '    return PrivateAttrImpl(private_attr, ge::AnyValue());',
    'OpImplKernelRegistry::OpImplFunctions& OpImplRegistry::': 'return types_to_impl_[op_type];',
    'OpImplRegistry::PrivateAttrList& OpImplRegistry::':
        '    static OpImplRegistry::PrivateAttrList emptyPrivateAttr;\n'
        '    return emptyPrivateAttr;',
    'OpImplKernelRegistry::PrivateAttrList& OpImplSpaceRegistry::':
        '    static OpImplKernelRegistry::PrivateAttrList emptyPrivateAttr;\n'
        '    return emptyPrivateAttr;',
    'std::map<OpImplRegistry::OpType, OpImplRegistry::OpImplFunctions>&': '    return types_to_impl_;',
    'OpImplRegister& OpImplRegister::': '    return *this;',
    'std::map<OpImplKernelRegistry::OpType, OpImplKernelRegistry::OpImplFunctions>&': '     return types_to_impl_;',
    'OpCtImplKernelRegistry::OpCtImplFunctions*': '    return nullptr;',
    'StreamMngFunc': '    return nullptr;',
    'OpImplRegistryHolderPtr': '    return nullptr;',
    'GetHiddenAddr': '    return nullptr;',
    'std::map<ge::AscendString, TuningTilingDefConstructor>&':
        '    static std::map<ge::AscendString, TuningTilingDefConstructor> instance;\n'
        '    return instance;',
    'std::shared_ptr<TuningTilingDef>': '    return nullptr;',
    'OpBankKeyConvertFun&': '    return convert_func_;',
    'OpBankParseFun&': '    return parse_func_;',
    'OpBankLoadFun&': '    return load_func_;',
    'std::unordered_map<ge::AscendString, OpBankKeyFuncInfoV2>&':
        '    static std::unordered_map<ge::AscendString, OpBankKeyFuncInfoV2> op_func_mapV2;\n'
        '    return op_func_mapV2;',
    'std::unordered_map<ge::AscendString, OpBankKeyFuncInfo>&':
        '    static std::unordered_map<ge::AscendString, OpBankKeyFuncInfo> op_func_map;'
        '    return op_func_map;',
    'OpBankLoadFunV2&': '    return load_funcV2_;',
    'OpBankParseFunV2&': '    return parse_funcV2_;',
    'OpBankKeyConvertFunV2&': '    return convert_funcV2_;',
    'StructSizeInfoBase& StructSizeInfoBase::': '    return *this;',
    'CTilingDataClassFactory& CTilingDataClassFactory::': '    return *this;',
    'TilingDataStructBase& TilingDataStructBase::': '    return *this;',
    'std::shared_ptr<TilingDef>': '    return nullptr;',
    'ByteBuffer&': '    return buf;',
    'ByteBuffer& OpRunInfo::': '    static ByteBuffer byte_buffer;\n    return byte_buffer;',
    'std::unordered_map<std::string, OpTilingFunc>& OpTilingRegistryInterf::':
        '    static std::unordered_map<std::string, OpTilingFunc> interf;\n    return interf;',
    'std::unordered_map<std::string, OpTilingFuncInfo>& OpTilingFuncRegistry::':
        '    static std::unordered_map<std::string, OpTilingFuncInfo> op_func_map;\n    return op_func_map;',
    'void* OpRunInfo::': '    return nullptr;',
    'ge::AscendString& OpCompileInfo::': '  static ge::AscendString compile_info;\n  return compile_info;',
    'std::unordered_map<std::string, OpTilingFuncV2>& OpTilingRegistryInterf_V2::':
        '    static std::unordered_map<std::string, OpTilingFuncV2> interf;\n    return interf;',
    'OpTilingFunc& OpTilingFuncInfo::': '    return this->tiling_func_;',
    'OpTilingFuncV2& OpTilingFuncInfo::': '    return this->tiling_func_v2_;',
    'OpTilingFuncV3& OpTilingFuncInfo::': '    return this->tiling_func_v3_;',
    'OpTilingFuncV4& OpTilingFuncInfo::': '    return this->tiling_func_v4_;',
    'OpParseFuncV3& OpTilingFuncInfo::': '    return this->parse_func_v3_;',
    'OpParseFuncV4& OpTilingFuncInfo::': '    return this->parse_func_v4_;',
    'ParseOpToGraphFunc': '    return nullptr;',
    'ParseSubgraphFunc': '    return nullptr;',
    'FusionParseParamByOpFunc': '    return nullptr;',
    'FusionParseParamFunc': '    return nullptr;',
    'ParseParamByOpFunc': '    return nullptr;',
    'AutoMappingSubgraphIOIndexFunc': '    return nullptr;',
    'ParseParamFunc': '    return nullptr;',
    'domi::FrameworkType': '    return FRAMEWORK_RESERVED;',
    'std::set<std::string> OpRegistrationData::': '    return {};',
    'domi::ImplyType': '    return domi::ImplyType::BUILDIN;',
    'OpRegistrationData& OpRegistrationData::': '    return *this;',
    'Promote& Promote::': '    return *this;',
    'Promote& PromoteImpl::': '    return obj;',
    'std::pair<GNodePtr, int32_t> GNode::':
        '    const std::pair<GNodePtr, int32_t> gnode_idx = {nullptr, 0xFF};\n    return gnode_idx;',
    'std::vector<int64_t>& OpRunInfo::': '    static std::vector<int64_t> vec;\n    return vec;',
    'std::vector<const char *>': '    return {};',
    'std::vector<FieldInfo>': '    return {};',
    'std::vector<GNodePtr>': '    return {};',
    'std::vector<std::pair<GNodePtr, int32_t>>': '    return {};',
    'std::vector<ge::NodePtr>': '    return {};',
    'std::vector<NodeToOutAnchor>': '    return {};',
    'std::vector<ConstGeTensorPtr>': '    return {};',
    'std::vector<GeTensorPtr>': '    return {};',
    'std::vector<ge::GeTensorDesc>': '    return {};',
    'std::vector<GNode>': '    return {};',
    'std::vector<GraphPtr>': '    return {};',
    'std::vector<std::vector<ShapeAndType>>&':
        '    static std::vector<std::vector<ShapeAndType>> sat;\n    return sat;',
    'std::vector<std::pair<int64_t, int64_t>>': '    return {};',
    'OpsProtoManager* OpsProtoManager::': '    static OpsProtoManager instance;\n    return &instance;',
    'Operator': '    return Operator();',
    'std::unique_ptr<BaseCustomOp>': 'return nullptr;',
    'std::map<size_t, std::pair<size_t, size_t>>': '    return {};',
    'GeTensor': '    return GeTensor();',
    'GeTensorPtr': '    return nullptr;',
    'ConstGeTensorPtr': '    return nullptr;',
    'ConstNodePtr': '    return nullptr;',
    'ComputeGraphPtr': '    return nullptr;',
    'ConstGeTensorBarePtr': '    return nullptr;',
    'GeTensorDesc': '    return GeTensorDesc();',
    'HcomTopoInfo& HcomTopoInfo::': '    static HcomTopoInfo hcom_topo_info;\n    return hcom_topo_info;',
    'HcomTopoInfo::TopoDescs*': '    return nullptr;',
    'AnchorInstanceInfo*': '    return nullptr;',
    'CompileTimeTensorDesc*': '    return nullptr;',
    'RuntimeAttrs*': '    return nullptr;',
    'DeviceTilingContextBuilder& DeviceTilingContextBuilder::': '    return *this;',
    'KernelContextHolder KernelRunContextBuilder::':
        'static KernelContextHolder default_holder;\n    return std::move(default_holder);',
    'ge::NodePtr KernelRunContextBuilder::': '    return nullptr;',
    'OpDescPtr': '    return nullptr;',
    'std::set<ge::AscendString>&': '    static std::set<ge::AscendString> as;\n    return as;',
    'ResourceContext*': '    return nullptr;',
    'std::unique_ptr<InferenceContext>': '    return nullptr;',
    'Shape ShapeAndType::': '    return Shape();',
    'NpuMemoryAllocator*': '    return nullptr;',
    'MemBlock*': '    return nullptr;',
    'rtStream_t': '    return nullptr;',
    'SingleOp*': '    return nullptr;',
    'DynamicSingleOp*': '    return nullptr;',
    'StreamResource*': '    return nullptr;',
    'std::unique_ptr<HybridDavinciModel>': '    return nullptr;',
    'ExecutorSubscribersScheduler& ModelV2Executor::': '    return subscribers_;',
    'ModelDesc& ModelV2Executor::': '    static ModelDesc default_model_desc;\n    return default_model_desc;',
    'std::unique_ptr<ge::Allocator>': '    return nullptr;',
    'ExecutorSubscriber& ExecutorSubscribersScheduler::': '    return subscriber_wrapper_;',
    'ge::Allocator*': '    return nullptr;',
    'ModelV2Executor*': '    return nullptr;',
    'Shape& ShapeRange::': '    return min_;',
    'Shape& ModelIoDesc::': '    return aipp_shape_;',
    'ShapeRange& ModelIoDesc::': '    return storage_shape_range_;',
    'OpImplSpaceRegistryArray*': '    return nullptr;',
    'OpImplSpaceRegistryV2Array*': '    return nullptr;',
    'OpImplKernelRegistry::OpImplFunctionsV2*': '    return nullptr;',
    'DefaultOpImplSpaceRegistryV2&': 'static DefaultOpImplSpaceRegistryV2 instance;\n    return instance;',
    'OpImplSpaceRegistryPtr&': '    static OpImplSpaceRegistryPtr null_ptr = nullptr;\n    return null_ptr;',
    'OpImplSpaceRegistryArray&': '    static OpImplSpaceRegistryArray space_registries;\n    return space_registries;',
    'ModelIoDesc*': '    return nullptr;',
    'OpLibRegistry& OpLibRegistry::': '    static OpLibRegistry instance;\n    return instance;',
    'SmallVector<int64_t, kDefaultDimsNum>& GeShape::GetMutableDims':
        '    static SmallVector<int64_t, 8U> vec;\n    return vec;',
    'GeShape& GeShape::operator=': '    return *this;',
    'GeShape& GeTensorDesc::': '    static GeShape ge_shape;\n    return ge_shape;',
    'std::vector<uint32_t> GeTensorDesc::': '    return {};',
    'GeTensorDesc& GeTensorDesc::operator=': '    return *this;',
    'ProtoAttrMap& GeTensorDesc::MutableAttrMap': '    static ProtoAttrMap attr;\n    return attr;',
    'ConstProtoAttrMap& GeTensorDesc::GetAttrMap': '    static ProtoAttrMap attr;\n    return attr;',
    'std::uint8_t*': '    return nullptr;',
    'std::shared_ptr<AlignedPtr>&': '    static std::shared_ptr<AlignedPtr> ap = nullptr;    return ap;',
    'TensorData& TensorData::operator=': '    return *this;',
    'GeTensorDesc& GeTensor::': '    static GeTensorDesc desc;\n    return desc;',
    'std::shared_ptr<AlignedPtr> GeTensor::': '    return nullptr;',
    'TensorData& GeTensor::': '    static TensorData data;\n    return data;',
    'GeTensor& GeTensor::operator=': '    return *this;',
    'std::unique_ptr<Graph>': '    return nullptr;',
    'std::vector<NodePtr>': '    return {};',
    'std::map<std::string, std::string>': '    return {};',
    'std::shared_ptr<const Node>': '    return nullptr;',
    'Shape TensorDesc::': '    return Shape();',
    'std::unique_ptr<uint8_t[], Tensor::DeleteFunc>': '    return nullptr;',
    'ge::Placement Tensor::': '    return static_cast<Placement>(0);',
    'ge::Placement TensorImpl::': '    return static_cast<Placement>(0);',
    'GeTensor* TensorAdapter::': '    return nullptr;',
    'HiddenInputsFuncRegistry& HiddenInputsFuncRegistry::GetInstance':
        '    static HiddenInputsFuncRegistry instance;\n    return instance;',
    'GetHiddenAddrs HiddenInputsFuncRegistry::': '    return nullptr;',
    'FollowType& OpParamDef::': '    static FollowType value = FollowType::INVALID_TYPE;\n    return value;',
    'DependScope& OpParamDef::': '    static DependScope value = DependScope::INVALID_SCOPE;\n    return value;',
    'gert::OpImplRegisterV2::TilingKernelFunc& OpAICoreDef::':
        '    static gert::OpImplRegisterV2::TilingKernelFunc func;\n    return func;',
    'gert::OpImplRegisterV2::InferShapeKernelFunc& OpDef::':
        '    static gert::OpImplRegisterV2::InferShapeKernelFunc func;\n    return func;',
    'gert::OpImplRegisterV2::InferShapeRangeKernelFunc& OpDef::':
        '    static gert::OpImplRegisterV2::InferShapeRangeKernelFunc func;\n    return func;',
    'gert::OpImplRegisterV2::InferDataTypeKernelFunc& OpDef::':
        '    static gert::OpImplRegisterV2::InferDataTypeKernelFunc func;\n    return func;',
    'std::vector<ge::AscendString>& OpDef::': '    static std::vector<ge::AscendString> vec;\n    return vec;',
    'std::map<ge::AscendString, OpDef::PortFollowInfo> OpDef::': '    return {};',
    'std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> OpDef::': '    return {};',
    'OpImplRegistry::OpImplFunctionsV2& OpImplRegistry::':
        '    static OpImplRegistry::OpImplFunctionsV2 func;\n    return func;',
    'OpImplRegistry::OpImplFunctionsV2* OpImplRegistry::': '    return nullptr;',
    'OpImplRegisterV2::PrivateAttrList& OpImplRegistry::':
        '    static OpImplRegisterV2::PrivateAttrList list;\n    return list;',
    'std::map<OpImplRegisterV2::OpType, OpImplRegistry::OpImplFunctionsV2>& OpImplRegistry::':
        '    static std::map<OpImplRegisterV2::OpType, OpImplRegistry::OpImplFunctionsV2> m;\n    return m;',
    'OpImplKernelRegistry::OpImplFunctionsV2* OpImplSpaceRegistry::': '    return nullptr;',
    'OpImplRegisterV2::PrivateAttrList& OpImplSpaceRegistry::':
        '    static OpImplRegisterV2::PrivateAttrList list;\n    return list;',
    'OpLibRegister& OpLibRegister::': '    return *this;',
    'std::unordered_map<std::string, OoInfo> OptionRegistry::': '    return {};',
    'OpParamDef OpDef::': '    static OpParamDef param("test");\n    return param;',
    'FormatCheckOption OpDef::': '    return FormatCheckOption::DEFAULT;',
    'OoInfo* OptionRegistry::': '    return nullptr;',
    'AscendString::AscendString(const char_t *const name)':
        r'''
          if (name != nullptr) {
            name_ = MakeShared<std::string>(name);
          }
        ''',
    'AscendString::AscendString(const char_t *const name, size_t length)':
        r'''
          if (name != nullptr) {
            name_ = MakeShared<std::string>(name, length);
          }
        ''',
    'const char_t* AscendString::GetString() const':
        r'''
          if (name_ == nullptr) {
            const static char *empty_value = "";
            return empty_value;
          }
          return (*name_).c_str();
        ''',
    'TensorDescValue& TensorDescValue::operator=': '    return *this;',
    'Shape': '    return Shape();',
    'AscendString TypeUtilsImpl::': '    return "";',
    'enum HcclServerType type, const char* soc = nullptr': 'enum HcclServerType type, const char* soc',
    'ge::AscendString soc_version = ""': 'ge::AscendString soc_version',
    'ops::HcclServerType': '    return HcclServerType::MAX;',
    'OppSoDesc& OppSoDesc::operator=': '    return *this;',
    'std::vector<ge::AscendString>': '    return {};',
}

# skip some private methods to Reduce the size of stub.so
SKIP_METHODS = [
    "MergeParam",
    "GetParamName",
    "GetParamType",
    "GetOriginDataTypes",
    "LoadOpsProtoSo",
    "LoadOpMasterSo",
    "LoadUpgradedOpsProtoSo",
    "LoadUpgradedOpMasterSo",
    "CreateOmOppDir",
    "RmOmOppDir",
    "OpTilingSinkRegister",
    "OpIsTilingSink",
    "GetHcclGroups",
    "GetHcclServerType",
    "GetIgnoreContiguous",
    "TilingParse",
    "LoadSoAndInitDefault",
    "AddCfgItem",
    "AddRegistry",
    "CreateOrGetOpImpl",
    "MergeTypesToImpl",
    "MergeFunctions",
    "MergeTypesToCtImpl",
    "MergeCtFunctions",
    "CopyGraphImpl",
    "PrintInOutTensorShape",
    "PostProcessAfterInfershape",
    "FeedStreamCtrlMap",
    "GenerateStreamCtrlMap",
    "ConvertPartitionCalledOp",
    "StridedOptimize",
    "GetAllCustomOpApiSoPaths",
    "CallInitFunc",
    "UpdateFormatImpl"
]

"""
    this attr is used for symbol table visible
"""
GE_ATTR = 'GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY'
VISIBILITY_ATTR = 'VISIBILITY_EXPORT'


"""
    max code len per line in hua_wei software programming specifications
"""
MAX_CODE_LEN_PER_LINE = 120

DEBUG = True

logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] [%(lineno)s] %(levelname)s: %(message)s',
                    level=logging.INFO)


def need_generate_func(func_line):
    """
    :param func_line:
    :return:
    """
    if func_line.strip().endswith("default") or func_line.strip().endswith("delete") \
            or func_line.strip().startswith("typedef") or func_line.strip().startswith("using"):
        return False
    return True


"""
    belows are patterns used for analyse .h file
"""
# pattern function
pattern_func = re.compile(r"""(^\s*)([a-zA-Z~_].*[)](?!.*{).*)(;.*)\n$""", re.VERBOSE | re.MULTILINE | re.DOTALL)
# pattern virtual function
pattern_virtual_func = re.compile(
    r"""^\s*virtual\s+(?:const\s+)?[:\w]+[ &*]+[:\w]+\([^()]*\)\s+(?:const\s+)?=\s+0;$""", re.VERBOSE)
# pattern comment
pattern_comment = re.compile(r'^\s*//')
pattern_comment_2_start = re.compile(r'^\s*/[*]')
pattern_comment_2_end = re.compile(r'[*]/\s*$')
# pattern visibility
pattern_visibility = re.compile(r'(FMK_FUNC_HOST_VISIBILITY|FMK_FUNC_DEV_VISIBILITY|VISIBILITY_EXPORT) *')
# pattern override
pattern_override = re.compile(r' +override\b')
# pattern weak
pattern_weak = re.compile(r' +__attribute__\(\(weak\)\)')
# pattern define
pattern_define = re.compile(r'^\s*#define')
pattern_define_return = re.compile(r'\\\s*$')
# pattern using
pattern_using = re.compile(r'^\s*using')
pattern_using_return = re.compile(r';\s*$')
# pattern include
pattern_include = re.compile(r'^\s*#include')
# blank line
pattern_blank_line = re.compile(r'^\s*$')
# virtual,explicit,friend,static
pattern_keyword = re.compile(r'(virtual\s+|explicit\s+|friend\s+|static\s+)')
# lead space
pattern_leading_space = re.compile(r'(^\s*)[a-zA-Z~_]')
# functions will have patterns such as func ( or func(
# but operator is an exception; the class name is preceded by an operator, and the above mode does not exist
# format like :"operator = ()"
pattern_func_name = re.compile(r'([a-zA-Z0-9~_\-]+\s*|operator?.*)[(]')
# template
pattern_template = re.compile(r'^\s*template')
pattern_template_end = re.compile(r'>\s*$')
# namespace
pattern_namespace = re.compile(r'namespace.*{')
# class : which can handle classA a and {not on the same line, but if found ';' after class,then don't deal with
pattern_class = re.compile(
    r'^\s*(class|struct)\s+((?:%s|%s)\s+)?([a-zA-Z0-9_\-]+<?)(?!.*;)' % (GE_ATTR, VISIBILITY_ATTR))
# pattern for function body start and end
pattern_start = re.compile('(?!namespace|class).+{')
pattern_end = re.compile('}')
# pattern for format func
pat_format_func = re.compile(
    r"""^((?:const\s+)?
    (?:[:\w]+
    |std::(?:vector|set|shared_ptr|unique_ptr)<[:\w* ]+>
    |std::(?:vector|set|shared_ptr|unique_ptr)<std::(?:vector|pair)<[:\w, ]+>>
    |std::(?:unique_ptr)<[:\w, \[\]]+>
    |std::(?:vector|set|shared_ptr)<std::string>
    |std::(?:map|unordered_map|pair)<[:\w]+[, ]+[:\w]+>
    |std::(?:map|unordered_map|pair)<[:\w]+,\s+std::pair<\w+,\s\w+>>
    |std::(?:map)<[:\w]+[, ]+std::(?:vector)<std::(?:pair)<[:\w, ]+>>>
    |(?:SmallVector)<[:\w]+[, ]+[:\w]+>
    ))
    (\s+)
    ([&*]?)""", re.VERBOSE)
# pattern for parsing ret_type & func_name
pat_search_func = re.compile(
    r"""^(?:const\s+)?
    (?P<ret_type>(?:[:\w]+
    |std::(?:vector|set|shared_ptr|unique_ptr)<[:\w* ]+>
    |std::(?:vector|set|shared_ptr|unique_ptr)<std::(?:vector|pair)<[:\w, ]+>>
    |std::(?:unique_ptr)<[:\w, \[\]]+>
    |std::(?:vector|set|shared_ptr)<std::string>
    |std::(?:map|unordered_map|pair)<[:\w]+[, ]+[:\w]+>
    |std::(?:map|unordered_map|pair)<[:\w]+,\s+std::pair<\w+,\s\w+>>
    |std::(?:map)<[:\w]+[, ]+std::(?:vector)<std::(?:pair)<[:\w, ]+>>>
    |(?:SmallVector)<[:\w]+[, ]+[:\w]+>
    )
    (?:[&*]+)?)\s+
    (?P<class_name>\w+)::\n?
    (?P<func_name>\w+|operator=)\s*\(""", re.VERBOSE)


class H2CC(object):
    def __init__(self, input_file, output_file, shared_includes_content):
        """
        :param input_file:
        :param output_file:
        :param shared_includes_content:
        """
        self.input_file = input_file
        file_dir = output_file[0:output_file.rindex('/')]
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        self.output_file = output_file
        self.shared_includes_content = shared_includes_content
        self.line_index = 0
        self.input_fd = open(self.input_file, 'r')
        self.input_content = self.input_fd.readlines()
        self.output_fd = open(self.output_file, 'w')

        # The state may be normal_now(in the middle of {}),class_now,namespace_now
        self.stack = []
        self.stack_class = []
        self.stack_template = []
        # record funcs generated by h2cc func
        self.func_list_exist = []

    def __del__(self):
        self.input_fd.close()
        self.output_fd.close()
        del self.stack
        del self.stack_class
        del self.stack_template
        del self.func_list_exist

    def just_skip(self):
        # skip blank line or comment
        if (pattern_blank_line.search(self.input_content[self.line_index])
                or pattern_comment.search(self.input_content[self.line_index])):
            self.line_index += 1
            return 'continue'
        # skip comment /* */
        elif pattern_comment_2_start.search(self.input_content[self.line_index]):
            while (self.line_index < len(self.input_content) and
                   not pattern_comment_2_end.search(self.input_content[self.line_index])):
                self.line_index += 1
            self.line_index += 1
            return 'continue'
        # skip #include
        elif pattern_include.search(self.input_content[self.line_index]):
            self.line_index += 1
            return 'continue'
        # skip define
        elif pattern_define.search(self.input_content[self.line_index]):
            while (pattern_blank_line.search(self.input_content[self.line_index])
                   or pattern_define_return.search(self.input_content[self.line_index])):
                self.line_index += 1
            self.line_index += 1
            return 'continue'
        # skip using
        elif pattern_using.search(self.input_content[self.line_index]):
            while (pattern_blank_line.search(self.input_content[self.line_index])
                   or not pattern_using_return.search(
                        self.input_content[self.line_index])):
                self.line_index += 1
            self.line_index += 1
            return 'continue'
        # skip extern const|constexpr type VARIABLE;
        elif re.compile(r'^.*\b(constexpr|const)\b.*(?= = ).*;$').search(self.input_content[self.line_index]):
            self.line_index += 1
            return 'continue'
        # static_assert
        elif re.compile(r'^\s*static_assert').search(self.input_content[self.line_index]):
            self.line_index += 1
            return 'continue'
        # skip virtual function
        elif pattern_virtual_func.search(self.input_content[self.line_index]):
            self.line_index += 1
            return 'continue'
        return 'pass'

    def handle_extern_variable(self):
        line = self.input_content[self.line_index]
        if re.search(r'^\s*extern\s+const.+(?!=).+;$', line):
            line = re.sub(r'^\s*extern\s*', '', line)
            default_value = 0
            if re.search(r'(std::)?(map|set|vector)', line):
                default_value = '{}'
            elif re.search(r'(std::)?string\b', line):
                default_value = '""'
            line = re.sub(r'^([^;]+)(;)$', fr'\1 = {default_value}\2\n', line)
            # line += ' = 0;\n'
            self.output_fd.write(line)
            self.line_index += 1
            return 'continue'
        return 'pass'

    def write_inc_content(self):
        for shared_include_content in self.shared_includes_content:
            self.output_fd.write(shared_include_content)
        self.output_fd.write("\n")

    def should_skip_func(self, func_line):
        return any(skip_method in func_line for skip_method in SKIP_METHODS)

    def h2cc(self):
        logging.info("start generate cc_file[%s] from h_file[%s]", self.output_file, self.input_file)
        self.write_inc_content()
        while self.line_index < len(self.input_content):
            # skip most of the lines that doesn't need to be considered
            if self.just_skip() == 'continue':
                continue

            if self.handle_namespace() == 'continue':
                continue

            # declaration extern variable
            if self.handle_extern_variable() == 'continue':
                continue

            # declaration class
            if self.handle_declaration_class() == 'continue':
                continue

            template_string = self.handle_template()

            line = self.input_content[self.line_index]
            if pattern_visibility.search(line):
                line = pattern_visibility.sub('', line)
            if pattern_override.search(line):
                line = pattern_override.sub('', line)
            if pattern_weak.search(line):
                line = pattern_weak.sub('', line)
            match_start = pattern_start.search(line)
            if self.handle_class(template_string, line, match_start) == "continue":
                continue

            # match }
            if self.handle_stack(match_start) == "continue":
                continue

            handle_func_result, line, start_i = self.handle_func(line)
            if handle_func_result == "continue":
                continue

            if self.should_skip_func(line):
                self.line_index += 1
                continue

            # here means func is found
            # delete key word
            line = pattern_keyword.sub('', line)
            # Class member function
            # if friend we will not add class name
            friend_match = re.search('friend ', line)

            if len(self.stack_class) > 0 and not friend_match:
                line, func_name = self.handle_class_member_func(line, template_string)
            # Normal functions
            else:
                line, func_name = self.handle_normal_func(line, template_string)

            need_generate = need_generate_func(line)

            # build func body
            line += self.implement_function(line)

            # comment
            # line = self.gen_comment(start_i) + line

            # write to out file
            self.write_func_content(line, func_name, need_generate)

            # next loop
            self.line_index += 1

        logging.info('Added %s functions', len(self.func_list_exist))
        logging.info('Successfully converted,please see ' + self.output_file)

    def handle_func(self, line):
        # determine which cases are not possible for a function
        match_left_bracket = re.search(r'(?<!.{7}\)|decltype)\((?!\*)', line)
        if not match_left_bracket:
            self.line_index += 1
            return "continue", line, None

        match_right_bracket = re.search(r'\)', line)
        start_i = self.line_index
        space_match = pattern_leading_space.search(line)
        # int abc(int a,\n int b)
        if match_left_bracket and (not match_right_bracket):
            self.line_index += 1
            break_line = self.input_content[self.line_index]
            if space_match:
                break_line = re.sub('^' + space_match.group(1), '', break_line)
            line += break_line

            while self.line_index < len(self.input_content):
                if (re.search(r'\)', break_line)
                        and not re.search(r'std::function<.+?> &input,', break_line)):
                    break
                self.line_index += 1
                break_line = self.input_content[self.line_index]
                break_line = re.sub('^' + space_match.group(1), '', break_line)
                line += break_line

        # begin match { }
        match_start = pattern_start.search(self.input_content[self.line_index])
        match_end = pattern_end.search(self.input_content[self.line_index])
        # like  ) {  or ) {}    int the last line
        if match_start:
            if not match_end:
                self.stack.append('normal_now')
            # ii = start_i
            # while ii <= self.line_index:
            #     ii += 1
            # skip { }, then continue
            # while not match_end:
            #     self.line_index += 1
            #     match_end = pattern_end.search(self.input_content[self.line_index])
            self.line_index += 1
            return "continue", line, start_i

        logging.info("line[%s]", line)
        # '  int abc();'->'int abc()'
        (line, match) = pattern_func.subn(r'\2\n', line)
        logging.info("line[%s]", line)
        if not match:
            self.line_index += 1
            return "continue", line, start_i

        # deal with case of 'return type' and 'func_name' are not in the same line, like: 'int \n abc(int a, int b)'
        pre_line = self.input_content[start_i - 1]
        if not pattern_visibility.search(pre_line) and re.search(r'^\s*(inline)?\s*[a-zA-Z0-9_]+\s*$', pre_line):
            line = pre_line + line
        line = line.lstrip()
        return "pass", line, start_i

    def handle_stack(self, match_start):
        if match_start:
            self.stack.append('normal_now')

        line = self.input_content[self.line_index]
        match_end = pattern_end.search(line)
        if match_end:
            top_status = self.stack.pop()
            if top_status == 'namespace_now':
                self.output_fd.write(line + '\n')
            elif top_status == 'class_now':
                self.stack_class.pop()
                self.stack_template.pop()
        if match_start or match_end:
            self.line_index += 1
            return "continue"

        if len(self.stack) > 0 and self.stack[-1] == 'normal_now':
            self.line_index += 1
            return "continue"
        return "pass"

    def handle_class(self, template_string, line, match_start):
        """
        :param template_string:
        :param line:
        :param match_start:
        :return:
        """
        match_class = pattern_class.search(line)
        if not match_class:
            return "pass"

        self.stack_template.append(template_string)
        self.stack.append('class_now')
        class_name = match_class.group(3)
        # class template specializations: class A<u,Node<u> >
        if '<' in class_name:
            k = line.index('<')
            fit = 1
            for ii in range(k + 1, len(line)):
                if line[ii] == '<':
                    fit += 1
                elif line[ii] == '>':
                    fit -= 1
                if fit == 0:
                    class_name += line[k + 1:ii + 1]
                    break
        logging.info('class_name[%s]', class_name)
        self.stack_class.append(class_name)

        # there could be a \n between class and {
        while not match_start:
            self.line_index += 1
            match_start = pattern_start.search(self.input_content[self.line_index])
        self.line_index += 1
        return "continue"

    def handle_declaration_class(self):
        line = self.input_content[self.line_index]
        match_result = re.search(r'^\s*class\s+(\w+);\s*$', line)
        if match_result:
            self.line_index += 1
            class_name = match_result.group(1)
            logging.info('declaration class:[%s]', class_name)
            # the class definition already exists
            if class_name in ['Graph', 'GNode', 'OpParamDefImpl', 'OpParamTrunk', 'OpAttrDefImpl',
                              'OpAICoreConfigImpl', 'OpAICoreDefImpl', 'OpAICPUDefImpl', 'OpHostCPUDefImpl', 'OpMC2DefImpl',
                              'OpDefImpl', 'Operator', 'GeTensorSerializeUtils', 'OpImplSpaceRegistryV2']:
                return 'pass'
            context = "class %s {\n"\
                      "  public:\n" \
                      "    %s() = default;\n" \
                      "    ~%s() = default;\n" \
                      "};\n\n" % (class_name, class_name, class_name)
            self.output_fd.write(context)
            return 'continue'
        return 'pass'

    def handle_template(self):
        line = self.input_content[self.line_index]
        match_template = pattern_template.search(line)
        template_string = ''
        if match_template:
            template_string = line
            while not pattern_template_end.search(line):
                self.line_index += 1
                line = self.input_content[self.line_index]
                template_string += line
            self.line_index += 1
        return template_string

    def handle_namespace(self):
        line = self.input_content[self.line_index]
        match_namespace = pattern_namespace.search(line)
        if match_namespace:
            self.output_fd.write(line + '\n')
            self.stack.append('namespace_now')
            self.line_index += 1
            return 'continue'
        return 'pass'

    def handle_normal_func(self, line, template_string):
        template_line = ''
        self.stack_template.append(template_string)
        if self.stack_template[-1] != '':
            template_line = re.sub(r'\s*template', 'template', self.stack_template[-1])
            # change '< class T = a, class U = A(3)>' to '<class T, class U>'
            template_line = re.sub(r'\s*=.*>(\s*)$', r'>\1', template_line)
            template_line = re.sub(r'\s*=.*,', ',', template_line)
            # template_line = re.sub(r'\s*=.*', '', template_line)

        line = re.sub(r'\s*=.*,', ',', line)
        line = re.sub(r'\s*=.*\)', ')', line)
        line = template_line + line
        self.stack_template.pop()

        func_name = re.search(r'^.*\)', line, re.MULTILINE | re.DOTALL).group()

        logging.info("line[%s]", line)
        logging.info("func_name[%s]", func_name)
        return line, func_name

    def handle_class_member_func(self, line, template_string):
        template_line = ''
        x = ''
        if template_string != '':
            template_string = re.sub(r'\s*template', 'template', template_string)
            template_string = re.sub(r'\s*=.*>(\s*)$', r'>\1', template_string)
            template_string = re.sub(r'\s*=.*,', ',', template_string)
            template_string = re.sub(r'\s*=.*', '', template_string)

        if self.stack_template[-1] != '':
            if not (re.search(r'<\s*>', self.stack_template[-1])):
                template_line = re.sub(r'^\s*template', 'template', self.stack_template[-1])
                if not (re.search(r'<.*>', self.stack_class[-1])):
                    # for x we get like template<class T, typename U> -> <T,U>
                    x = re.sub(r'template\s*<', '<', template_line)  # remove template -> <class T, typename U>
                    x = re.sub(r'\n', '', x)
                    x = re.sub(r'\s*=.*,', ',', x)
                    x = re.sub(r'\s*=.*>', '>', x)
                    x = x.rstrip()  # remove \n

                    # remove class,typename ->  <T, U>
                    x = re.sub(r'(class|typename)\s+|(<class>|<typename>\s*class)', '', x)
                    x = re.sub(r'<\s+', '<', x)
                    x = re.sub(r'\s+>', '>', x)
                    x = re.sub(r'\s+,', ',', x)
                    x = re.sub(r',\s+', ', ', x)

        line = re.sub(r'\s*=\s+0', '', line)
        line = re.sub(r'\s*=\s+.*,', ',', line)
        line = re.sub(r'\s*=\s+.*\)', ')', line)
        logging.info("x[%s]\nline[%s]", x, line)

        # if the function is long, void ABC::foo()
        # breaks into two lines void ABC::\n foo()
        rep_fmt = '%s%s::{}%s' % (self.stack_class[-1], x, r'\1(')

        temp_line = pattern_func_name.sub(rep_fmt.format(''), line, count=1)

        if len(temp_line) > MAX_CODE_LEN_PER_LINE:
            line = pattern_func_name.sub(rep_fmt.format('\n'), line, count=1)
        else:
            line = temp_line

        logging.info("line[%s]", line)

        # add template as the above if there is one
        template_line = re.sub(r'\s*=.*>(\s*)$', r'>\1', template_line)
        template_line = re.sub(r'\s*=.*,', ',', template_line)
        template_line = re.sub(r'\s*=.*', '', template_line)
        line = template_line + template_string + line
        func_name = re.search(r'^.*\)', line, re.MULTILINE | re.DOTALL).group()
        line = re.sub(r'\b(KernelInfo)\b', r'KernelRegistry::\1', line)
        line = re.sub(r'\b(KernelFuncs)\b', r'KernelRegistry::\1', line)
        line = re.sub(r'(?<!::)\b(OpImplFunctions)\b', r'OpImplKernelRegistry::\1', line)
        line = re.sub(r'(?<!::)\b(OpType)\b', r'OpImplKernelRegistry::\1', line)
        line = re.sub(r'\b(PrivateAttrList &OpImplKernelRegistry::)', r'OpImplKernelRegistry::\1', line)
        line = re.sub(r'\b(TopoDescs)\b', r'HcomTopoInfo::\1', line)
        # add inner class type

        logging.info("line[%s]", line)
        logging.info("func_name[%s]", func_name)
        return line, func_name

    def write_func_content(self, content, func_name, need_generate):
        if not (func_name in self.func_list_exist) and need_generate:
            self.output_fd.write(content)
            self.func_list_exist.append(func_name)
            logging.info('add func:[%s]', func_name)

    def gen_comment(self, start_i):
        comment_line = ''
        # Function comments are on top of function declarations, copy them over
        k = start_i - 1  # one line before this func start
        if pattern_template.search(self.input_content[k]):
            k -= 1
        # 多行注释的尾注释符
        if pattern_comment_2_end.search(self.input_content[k]):
            comment_line = self.input_content[k].lstrip()
            while not pattern_comment_2_start.search(self.input_content[k]):
                k -= 1
                comment_line = self.input_content[k].lstrip() + comment_line
        else:
            for j in range(k, 0, -1):
                c_line = self.input_content[j]
                # 行注释符
                if pattern_comment.search(c_line):
                    c_line = re.sub(r'\s*//', '//', c_line)  # 把前面的缩进去掉
                    comment_line = c_line + comment_line
                else:
                    break
        return comment_line

    @staticmethod
    def get_return_statements(func):
        # int *&Foo() -> int*& Foo()
        func = pat_format_func.sub(r'\1\3\2', func)
        if func.strip() in RETURN_STATEMENTS:
            logging.info('func:[%s] matched!', func.strip())
            return re.sub(r'^ {8}', '', RETURN_STATEMENTS[func.strip()], count=0, flags=re.MULTILINE).strip('\n')

        m = pat_search_func.search(func)
        if not m:
            return None
        logging.info('ret_type: %s, class_name: %s, func_name: %s', *m.group('ret_type', 'class_name', 'func_name'))

        type_cls_func_name = '%s %s::%s' % m.group('ret_type', 'class_name', 'func_name')
        if type_cls_func_name in RETURN_STATEMENTS:
            logging.info('type_cls_func_name:[%s] matched!', type_cls_func_name)
            return RETURN_STATEMENTS[type_cls_func_name]

        type_cls_name = '%s %s::' % m.group('ret_type', 'class_name')
        if type_cls_name in RETURN_STATEMENTS:
            logging.info('type_cls_name:[%s] matched!', type_cls_name)
            return RETURN_STATEMENTS[type_cls_name]

        type_only = m.group('ret_type')
        if type_only in RETURN_STATEMENTS:
            logging.info('type_only:[%s] matched!', type_only)
            return RETURN_STATEMENTS[type_only]
        return None

    @staticmethod
    def handle_template_func_def(func, function_def):
        # 处理模板函数的情况
        # 对于模板函数，我们需要从函数签名中提取返回类型
        # 例如: template<typename ArgType> graphStatus GNode::GetAttrImpl(...)
        if not func.strip().startswith('template<'):
            return False, function_def
        has_been_handled = False
        # 提取返回类型，跳过template声明
        lines = func.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('template<'):
                continue
            # 找到返回类型和函数名
            if re.search(r'^\w+\s+\w+::', line):
                # 例如: graphStatus GNode::
                parts = line.split()
                if len(parts) >= 2:
                    return_type = parts[0]
                    if RETURN_STATEMENTS.__contains__(return_type):
                        function_def += RETURN_STATEMENTS[return_type]
                        logging.info("Template func get return type:%s", return_type)
                        has_been_handled = True
                        break
        if not has_been_handled:
            # 如果没有找到合适的返回类型，使用默认的graphStatus
            if 'graphStatus' in func:
                function_def += RETURN_STATEMENTS['graphStatus']
                logging.info("Template func using default return type: graphStatus")
                has_been_handled = True
            else:
                logging.info("Unhandled template func:%s", func)
                logging.warning("Cannot determine return type for template function")
        return has_been_handled, function_def

    @staticmethod
    def implement_function(func):
        function_def = ''
        # 特殊情况: 类成员变量是引用类型
        if func.strip() == 'OpImplRegister::OpImplRegister(const ge::char_t *op_type)':
            function_def += '    : functions_(OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type))\n'

        function_def += '{\n'

        # function_def += '    printf("[%s:%d][STUB]%s", __FILE__, __LINE__, __FUNCTION__);\n'

        return_statements = H2CC.get_return_statements(func)

        if return_statements is not None:
            function_def += return_statements
        else:
            # 处理模板函数的情况
            template_result = H2CC.handle_template_func_def(func, function_def)
            if template_result[0]:
                function_def = template_result[1]
            elif not re.search(r'::\w+(?=\()', func):
                all_items = func.split()
                start = 0
                return_type = all_items[start]
                if return_type == 'extern':
                    return_type = all_items[2]
                if return_type == "const":
                    start += 1
                    return_type = all_items[start]
                if return_type.startswith(('std::map', 'std::set', 'std::vector')):
                    # return_type = "std::map"
                    return_type = return_type[0:return_type.index("<")]
                if return_type.endswith('*') \
                        or (len(all_items) > start + 1 and all_items[start + 1].startswith('*')) \
                        or return_type.startswith('std::unique_ptr') \
                        or return_type.startswith('std::shared_ptr'):
                    return_type = "Ptr"
                if len(all_items) > start + 1 and all_items[start + 1].startswith('&'):
                    return_type += "&"

                if RETURN_STATEMENTS.__contains__(return_type):
                    function_def += RETURN_STATEMENTS[return_type]
                else:
                    logging.info("Unhandled func:%s", func)
                    logging.warning("Unhandled return type:%s", return_type)

        function_def += '\n'
        function_def += '}\n'
        function_def += '\n'
        return function_def


def collect_header_files(inc_file):
    shared_includes_content = []
    inc_dirs = [
        'inc/external/',
        'inc/framework/',
        'inc/',
        'runtime/v1/',
        'metadef/base/',
        'base/',
        'metadef/register/',
        'metadef/third_party/transformer/src/',
        'include/register/',
        'graph_metadef/'
        # add include dirs
    ]
    for inc_dir in inc_dirs:
        if inc_file.find(inc_dir) != -1:
            include_str = '#include "{}"\n'.format(inc_file[inc_file.index(inc_dir) + len(inc_dir):])
            shared_includes_content.append(include_str)
            break
    else:
        logging.error('please check inc_dirs')
    shared_includes_content.append('#include <iostream>\n')

    need_any_value_headers = ['op_impl_registry_base.h',
                              'op_impl_registry_holder_manager.h']

    if inc_file.endswith('op_def.h'):
        shared_includes_content.append('#include "opdef/op_def_impl.h"\n')
    elif inc_file.endswith('operator.h'):
        shared_includes_content.append('#include "graph/graph.h"\n')
    elif inc_file.endswith('ascend_string.h'):
        shared_includes_content.append('#include "common/util/mem_utils.h"\n')
    elif len([var for var in need_any_value_headers if inc_file.endswith(var)]) > 0:
        shared_includes_content.append('#include "graph/any_value.h"\n')

    return shared_includes_content


def generate_stub_file(inc_file, out_cc_dir):
    """
    :param inc_file:
    :param out_cc_dir:
    :return:
    """
    shared_includes_content = collect_header_files(inc_file)
    cc_file = re.sub(r'([^/]+)\.h$', r'stub_\1.cc', inc_file)
    h_2_cc = H2CC(inc_file, out_cc_dir + cc_file[cc_file.rindex('/') + 1:], shared_includes_content)
    h_2_cc.h2cc()


def gen_code(inc_files, out_cc_dir):
    """
    :param inc_files:
    :param out_cc_dir:
    :return:
    """
    if not out_cc_dir.endswith('/'):
        out_cc_dir += '/'
    for inc_file in inc_files:
        if not os.path.isabs(inc_file):
            logging.warning('inc_file:[%s] not absolute path, and will be ignored.', inc_file)
            continue
        generate_stub_file(inc_file, out_cc_dir)


def main():
    if len(sys.argv) < 3:
        logging.error("script %s must have 2 input parameters!", sys.argv[0])
        return
    out_cc_dir = sys.argv[1]
    inc_files = sys.argv[2:]
    gen_code(inc_files, out_cc_dir)


if __name__ == '__main__':
    main()

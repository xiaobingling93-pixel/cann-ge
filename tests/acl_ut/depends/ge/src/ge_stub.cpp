/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <string>
#include <securec.h>
#include "framework/common/ge_format_util.h"
#include "framework/executor/ge_executor.h"
#include "framework/generator/ge_generator.h"
#include "framework/common/profiling_definitions.h"
#include "framework/runtime/gert_api.h"
#include "framework/runtime/model_desc.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/stream_executor.h"
#include "framework/runtime/mem_allocator.h"
#include "framework/memory/allocator_desc.h"

#include "framework/common/util.h"
#include "common/helper/om_file_helper.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_tensor.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/tensor.h"
#include "graph/model.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/ge_attr_value.h"
#include "graph/operator.h"
#include "ge/ge_api.h"
#include "common/ge_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/opsproto_manager.h"
#include "graph/operator_factory.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "aprof_pub.h"
#include "graph/detail/attributes_holder.h"
#include "acl_stub.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "platform/platform_info.h"
#include "nlohmann/json.hpp"
#include "register/stream_manage_func_registry.h"
#include "base/err_mgr.h"

using namespace ge;

namespace {
    std::unordered_map<TypeId, AnyValue::ValueType> type_ids_to_value_type = {
    {nullptr, AnyValue::VT_NONE},
    {GetTypeId<std::string>(), AnyValue::VT_STRING},
    {GetTypeId<float>(), AnyValue::VT_FLOAT},
    {GetTypeId<bool>(), AnyValue::VT_BOOL},
    {GetTypeId<int64_t>(), AnyValue::VT_INT},
    {GetTypeId<GeTensorDesc>(), AnyValue::VT_TENSOR_DESC},
    {GetTypeId<GeTensor>(), AnyValue::VT_TENSOR},
    {GetTypeId<Buffer>(), AnyValue::VT_BYTES},
    {GetTypeId<proto::GraphDef>(), AnyValue::VT_GRAPH},
    {GetTypeId<NamedAttrs>(), AnyValue::VT_NAMED_ATTRS},
    {GetTypeId<std::vector<std::vector<int64_t>>>(), AnyValue::VT_LIST_LIST_INT},
    {GetTypeId<DataType>(), AnyValue::VT_DATA_TYPE},
    {GetTypeId<std::vector<std::vector<float>>>(), AnyValue::VT_LIST_LIST_FLOAT},
    {GetTypeId<std::vector<std::string>>(), AnyValue::VT_LIST_STRING},
    {GetTypeId<std::vector<float>>(), AnyValue::VT_LIST_FLOAT},
    {GetTypeId<std::vector<bool>>(), AnyValue::VT_LIST_BOOL},
    {GetTypeId<std::vector<int64_t>>(), AnyValue::VT_LIST_INT},
    {GetTypeId<std::vector<GeTensorDesc>>(), AnyValue::VT_LIST_TENSOR_DESC},
    {GetTypeId<std::vector<GeTensor>>(), AnyValue::VT_LIST_TENSOR},
    {GetTypeId<std::vector<Buffer>>(), AnyValue::VT_LIST_BYTES},
    {GetTypeId<std::vector<proto::GraphDef>>(), AnyValue::VT_LIST_GRAPH},
    {GetTypeId<std::vector<NamedAttrs>>(), AnyValue::VT_LIST_NAMED_ATTRS},
    {GetTypeId<std::vector<DataType>>(), AnyValue::VT_LIST_DATA_TYPE},
};
    std::map<std::string, std::string> ge_thread_global_options;
    std::map<std::string, std::string> ge_context_option_name_map;
    int32_t ge_context_stream_sync_timeout = -1;
    int32_t ge_context_event_sync_timeout = -1;
}

ge::Status aclStub::SetDump(const ge::DumpConfig &dumpConfig)
{
    return ge::SUCCESS;
}

Status aclStub::GEInitialize(const std::map<AscendString, AscendString>& options)
{
    return SUCCESS;
}

Status aclStub::Finalize()
{
    return SUCCESS;
}

Status aclStub::Ge_Generator_Finalize()
{
    return SUCCESS;
}

Status aclStub::GEFinalize()
{
    return SUCCESS;
}

Status aclStub::BuildSingleOpModel(ge::OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                   const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                   int32_t compile_flag, ModelBufferData &model_buff)
{
    return SUCCESS;
}

Status aclStub::BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                            int32_t compile_flag, ModelBufferData &model_buff,
                            GraphStage graph_stage, ComputeGraphPtr &compute_graph)
{
    return SUCCESS;
}

graphStatus aclStub::SetShapeRange(const std::vector<std::pair<int64_t,int64_t>> &range)
{
    return GRAPH_SUCCESS;
}

bool aclStub::ReadBytesFromBinaryFile(char const *file_name, char **buffer, int &length)
{
    return true;
}

Status aclStub::Initialize(const std::map<std::string, std::string> &options)
{
    return SUCCESS;
}

Status aclStub::Initialize(const std::map<std::string, std::string> &options, OmgContext &omgContext)
{
    return SUCCESS;
}

Status aclStub::LoadSingleOpV2(const std::string &modelName,
                                    const ModelData &modelData,
                                    void *stream,
                                    SingleOp **single_op,
                                    const uint64_t model_id)
{
    return SUCCESS;
}

Status aclStub::SetAllocator(void *const stream, ge::Allocator *const external_allocator)
{
    return SUCCESS;
}

Status aclStub::LoadDynamicSingleOpV2(const std::string &model_name,
                            const ge::ModelData &modelData,
                            void *stream,
                            DynamicSingleOp **single_op,
                            const uint64_t model_id)
{
    return SUCCESS;
}

Status aclStub::ExecuteAsync(DynamicSingleOp *executor,
                    const std::vector<GeTensorDesc> &input_desc,
                    const std::vector<DataBuffer> &inputs,
                    std::vector<GeTensorDesc> &output_desc,
                    std::vector<DataBuffer> &outputs)
{
    return SUCCESS;
}

Status aclStub::ExecuteAsync(SingleOp *executor,
                            const std::vector<DataBuffer> &inputs,
                            std::vector<DataBuffer> &outputs)
{
    return SUCCESS;
}

bool aclStub::GetBool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, bool &value)
{
    return true;
}

bool aclStub::GetInt(ge::AttrUtils::ConstAttrHolderAdapter&& obj, const std::string &name, int32_t &value)
{
    return true;
}

bool aclStub::GetListNamedAttrs(ge::AttrUtils::ConstAttrHolderAdapter &&obj, std::string const &name, vector<AnyValue::NAMED_ATTRS> &value)
{
    return true;
}

std::map<string, AnyValue> aclStub::GetAllAttrs()
{
    AnyValue attr;
    std::string name = "ATTR_MODEL_test";
    std::map<string, AnyValue> m;
    m.insert(std::make_pair(name, attr));
    return m;
}

std::string aclStub::RealPath(const char *path)
{
    return "test";
}

graphStatus aclStub::GetOpsTypeList(std::vector<ge::AscendString> &all_ops)
{
    return 0;
}

Status aclStub::GetModelDescInfo(uint32_t modelId, std::vector<TensorDesc>& inputDesc,
                                 std::vector<TensorDesc>& outputDesc, bool new_model_desc)
{   
    return SUCCESS;
}

Status aclStub::GetModelDescInfoFromMem(const ModelData &model_data, ModelInOutInfo &info)
{
    return SUCCESS;
}

graphStatus aclStub::GetShapeRange(std::vector<std::pair<int64_t,int64_t>> &range)
{
    return 0;
}

Format aclStub::GetFormat()
{
    Format format = FORMAT_NCHW;
    return format;
}

Status aclStub::GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                    int32_t &dynamic_type)
{
    batch_info.push_back({1});
    return SUCCESS;
}

Status aclStub::LoadModelFromData(uint32_t &model_id, const ModelData &modelData,
                                void *dev_ptr, size_t memsize, void *weight_ptr, size_t weightsize)
{
    static uint32_t cnt = 0;
    ++cnt;
    model_id = cnt;
    return SUCCESS;
}

Status aclStub::LoadModelFromDataWithArgs(uint32_t &model_id, const ModelData &model_data, const ModelLoadArg &load_arg)
{
  static uint32_t cnt = 0;
  ++cnt;
  model_id = cnt;
  return SUCCESS;
}

graphStatus aclStub::LoadDataFromFile(std::string const &path, ModelData &modelData)
{
    return SUCCESS;
}

ge::graphStatus aclStub::LoadDataFromFileV2(const char *path, ge::ModelData &model_data)
{
    return GRAPH_SUCCESS;
}

void gert::RtSession::DestroyResources() const {
  return;
}

std::unique_ptr<gert::ModelV2Executor> aclStub::LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                                          ge::graphStatus &error_code)
{
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    error_code = GRAPH_SUCCESS;
    return executor;
}

std::unique_ptr<gert::ModelV2Executor> aclStub::LoadExecutorFromModelDataWithRtSession(const ge::ModelData &model_data,
                                                                              gert::RtSession *const rt_session,
                                                                              ge::graphStatus &error_code)

{
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    error_code = GRAPH_SUCCESS;
    return executor;
}
std::unique_ptr<gert::ModelV2Executor>
aclStub::LoadExecutorFromModelDataWithMem(const ge::ModelData &model_data, ge::graphStatus &error_code,
                                          const void *weight_ptr, const size_t weight_size) {
  return aclStub::LoadExecutorFromModelData(model_data, error_code);
}

std::unique_ptr<gert::StreamExecutor> aclStub::LoadStreamExecutorFromModelData(const ge::ModelData &model_data, const void *weight_ptr,
                                                                const size_t weight_size, ge::graphStatus &error_code)
{
    std::unique_ptr<gert::StreamExecutor> executor = std::unique_ptr<gert::StreamExecutor>(new gert::StreamExecutor(nullptr));
    error_code = ge::GRAPH_SUCCESS;
    return executor;
}

std::unique_ptr<gert::StreamExecutor> aclStub::LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                               const gert::LoweringOption &optimize_option,
                                                                               ge::graphStatus &error_code)
{
    std::unique_ptr<gert::StreamExecutor> executor = std::unique_ptr<gert::StreamExecutor>(new gert::StreamExecutor(nullptr));
    error_code = ge::GRAPH_SUCCESS;
    return executor;
}

ge::graphStatus aclStub::IsDynamicModel(const char *file_path, bool &is_dynamic_model)
{
    is_dynamic_model = true;
    return SUCCESS;
}

Status aclStub::LoadModelWithQ(uint32_t &model_id, const ge::ModelData &ge_model_data,
                const std::vector<uint32_t> &input_queue_ids, const std::vector<uint32_t> &output_queue_ids)
{
    return SUCCESS;
}

Status aclStub::LoadModelWithQ(uint32_t &model_id, const ge::ModelData &ge_model_data,
                const ge::ModelQueueArg &queue_arg)
{
    return SUCCESS;
}

Status aclStub::UnloadModel(uint32_t modelId)
{
    return SUCCESS;
}

ge::Status aclStub::LoadOm2DataFromFile(const std::string &model_path, ge::ModelData &model_data) {
  return SUCCESS;
}
std::unique_ptr<gert::Om2ModelExecutor> aclStub::LoadOm2ExecutorFromData(ge::ModelData &model_data, ge::Status &error_code) {
  return nullptr;
}
ge::Status aclStub::IsOm2Model(const void *data, size_t size, bool &is_support) {
  return SUCCESS;
}
ge::Status aclStub::IsOm2Model(const char *file_path, bool &is_support) {
  return SUCCESS;
}
ge::Status aclStub::GetModelDescInfo(std::vector<ge::TensorDesc> &input_desc, std::vector<ge::TensorDesc> &output_desc,
                                     bool new_model_desc) {
  return SUCCESS;
}

Status aclStub::GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size)
{
    return SUCCESS;
}

Status aclStub::GetMemAndWeightSize(const void *model_data, size_t model_size, size_t &mem_size, size_t &weight_size)
{
    return SUCCESS;
}

Status aclStub::ExecModel(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                            const std::vector<ge::GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                            std::vector<ge::GeTensorDesc> &output_desc, bool async_mode)
{
    ge::GeTensorDesc geDescTmp;
    output_desc.push_back(geDescTmp);
    return SUCCESS;
}

Status aclStub::SetDynamicBatchSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t batch_size)
{
    return SUCCESS;
}

Status aclStub::SetDynamicImageSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t image_height, uint64_t image_width)
{
    return SUCCESS;
}

Status aclStub::SetDynamicDims(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                    const vector<uint64_t> &dynamic_dims)
{
    return SUCCESS;
}

Status aclStub::GetCurDynamicDims(uint32_t model_id, const vector<uint64_t> &dynamic_dims,
                                        vector<uint64_t> &cur_dynamic_dims)
{
    return SUCCESS;
}

Status aclStub::GetAippType(uint32_t model_id, uint32_t index, ge::InputAippType &type, size_t &aippindex)
{
    type = ge::DATA_WITH_DYNAMIC_AIPP;
    aippindex = 3;
    return SUCCESS;
}

Status aclStub::GetUserDesignateShapeOrder(uint32_t model_id, vector<string> &user_designate_shape_order)
{
    return SUCCESS;
}

ge::Status aclStub::GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type)
{
    batch_info.push_back(1);
    return SUCCESS;
}

Status aclStub::GetModelAttr(uint32_t model_id,std::vector<std::string> &dynamic_output_shape_info)
{
    dynamic_output_shape_info.push_back({"0:0:1,3,224,224"});
    return SUCCESS;
}

Status aclStub::GetOpAttr(uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                   std::string &attr_value)
{
    return SUCCESS;
}

 graphStatus aclStub::GetName(AscendString &name)
 {
     return 0;
 }

Status aclStub::GetAIPPInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_params)
{
    return SUCCESS;
}

Status aclStub::GetAippInfo(const uint32_t index, ge::AippConfigInfo &aipp_info)
{
  return SUCCESS;
}

Status aclStub::GetAippType(const uint32_t index, ge::InputAippType &aipp_type, size_t &aipp_index)
{
    return SUCCESS;
}

Status aclStub::GetBatchInfoSize(uint32_t model_id, size_t &shape_count)
{
    return SUCCESS;
}

Status aclStub::GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &origOutputInfo)
{
    return SUCCESS;
}

Status aclStub::GetOriginAippInputInfo( uint32_t index, OriginInputInfo &origOutputInfo)
{
    return SUCCESS;
}

Status aclStub::GetAllAippInputOutputDims(uint32_t model_id, uint32_t index,
                                        std::vector<InputOutputDims> &input_dims,
                                        std::vector<InputOutputDims> &output_dims)
{
    return SUCCESS;
}

Status aclStub::GetAllAippInputOutputDims(uint32_t index,
                                        std::vector<InputOutputDims> &input_dims,
                                        std::vector<InputOutputDims> &output_dims)
{
    return SUCCESS;
}

Status aclStub::Init(uint8_t *model_data, const uint32_t model_data_size)
{
    return MockFunctionTest::aclStubInstance().Init(model_data, model_data_size);
}

Status aclStub::SetDynamicAippData(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                        const std::vector<kAippDynamicBatchPara> &aippBatchPara,
                                        const kAippDynamicPara &aippParms)
{
    return SUCCESS;
}

int aclStub::Init()
{
    return 0;
}

bool aclStub::OpsProtoManager_Initialize(const std::map<std::string, std::string> &options)
{
    return true;
}

Status aclStub::TransShape(const TensorDesc &src_desc,
                                Format dst_format,
                                std::vector<int64_t> &dst_shape)
{
    return SUCCESS;
}

Status aclStub::GetModelPartition(ModelPartitionType type, ModelPartition &partition)
{
    return SUCCESS;
}

graphStatus aclStub::Load(const uint8_t *data, size_t len, Model &model)
{
    return 0;
}

bool aclStub::HasAttr(AttrUtils::ConstAttrHolderAdapter&& obj, const string &name)
{
    return true;
}

bool aclStub::GetListTensor(AttrUtils::ConstAttrHolderAdapter&& obj, const string& name, vector<ConstGeTensorPtr>& value)
{
    return true;
}

bool aclStub::IsOriginShapeInRange(const gert::Shape &shape) {
    return true;
}

gert::ModelV2Executor *aclStub::GetOrCreateLoaded(rtStream_t stream, const gert::ModelExecuteArg &arg)
{
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    executor->Load(arg);
    return executor.get();
}

gert::ModelV2Executor *aclStub::CreateAndLoad(rtStream_t stream, const gert::ModelExecuteArg &arg)
{
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
    executor->Load(arg);
    return executor.get();
}

ge::graphStatus aclStub::Erase(rtStream_t stream)
{
    return ge::GRAPH_SUCCESS;
}


std::unique_ptr<gert::ModelV2Executor> aclStub::LoadExecutorFromFile(const char *file_path, ge::graphStatus &error_code)
{
    return std::unique_ptr<gert::ModelV2Executor>(new(std::nothrow) gert::ModelV2Executor);
}

ge::graphStatus aclStub::Load()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus aclStub::Load(const gert::ModelExecuteArg &arg)
{
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus aclStub::Load(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg) {
    return ge::GRAPH_SUCCESS;
}
std::unique_ptr<ge::Allocator> aclStub::Create(const gert::TensorPlacement &placement)
{
    return nullptr;
}

ge::graphStatus aclStub::Execute(const gert::ModelExecuteArg &arg,
                        gert::Tensor **inputs, size_t input_num,
                        gert::Tensor **outputs, size_t output_num)
{
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus aclStub::ExecuteSync(gert::Tensor **inputs, size_t input_num,
                            gert::Tensor **outputs, size_t output_num)
{
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus aclStub::UnLoad()
{
    return ge::GRAPH_SUCCESS;
}

uint32_t aclStub::InitializePlatformInfo()
{
    return 0;
}

uint32_t aclStub::GetPlatformInfos(
    const std::string SoCVersion, fe::PlatFormInfos &platformInfo, fe::OptionalInfos &optionalInfo)
{
    return 0;
}

uint32_t aclStub::InitRuntimePlatformInfos(const std::string &SoCVersion)
{
    return 0;
}

uint32_t aclStub::GetRuntimePlatformInfosByDevice(const uint32_t &device_id, fe::PlatFormInfos &platform_infos) {
    return 0;
}

bool aclStub::GetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res) {
    return true;
}

bool aclStub::GetPlatformResWithLock(const string &label, const string &key, string &val)
{
    return true;
}

uint32_t aclStub::UpdateRuntimePlatformInfosByDevice(const uint32_t &device_id, fe::PlatFormInfos &platform_infos) {
    return 0;
}

MockFunctionTest::MockFunctionTest()
{
    ResetToDefaultMock();
}

MockFunctionTest& MockFunctionTest::aclStubInstance()
{
    static MockFunctionTest stub;
    return stub;
};

void  MockFunctionTest::ResetToDefaultMock() {
}

namespace ge {
const std::string ATTR_NAME_STORAGE_FORMAT = "storage_format";
const std::string ATTR_NAME_STORAGE_SHAPE = "storage_shape";
const std::string ATTR_NAME_UNREGST_OPPATH = "_unregst_oppath";
const std::string ATTR_NAME_UNREGST_ATTRLIST = "_unregst _attrlist";
const std::string ATTR_NAME_DYNAMIC_INPUT_START = "_dynamic_input_index_start";
const std::string ATTR_NAME_DYNAMIC_INPUT_END = "_dynamic_input_index_end";
const std::string ATTR_NAME_WEIGHTS = "value";
const std::string CONST_ATTR_NAME_INPUT = "is_const";
const std::string ATTR_NAME_FUZZ_BUILD_RES_ATTRS = "_fuzz_build_res";
const std::string ATTR_NAME_PLACEMENT = "_mem_type";
const std::string ATTR_NAME_FUZZ_INPUTS_SUPPORTED_ATTRS = "_inputs_support_info";
const std::string ATTR_NAME_FUZZ_OUTPUTS_SUPPORTED_ATTRS = "_outputs_support_info";
const std::string ATTR_NAME_BUILD_MODE = "_build_mode";
const std::string ATTR_NAME_VALUE = "_value";
const std::string ATTR_NAME_VALUE_RANGE = "_value_range";
const std::string ATTR_NAME_IS_DYNAMIC_MODEL = "_is_dynamic_model";
const std::string ATTR_SINGLE_OP_SCENE = "_single_op_scene";


namespace {
bool g_geAttrValueBool;
std::string g_geAttrValueString;
float g_geAttrValueFloat;
DataType g_geAttrValueDataType;
int64_t g_geAttrValueInt;
thread_local GEThreadLocalContext threadContext;
GEContext geContext;
AnyValue g_geAttrValue;

std::vector<bool> g_geAttrValueListBool;
std::vector<std::string> g_geAttrValueListString;
std::vector<float> g_geAttrValueListFloat;
std::vector<int64_t> g_geAttrValueListInt;
std::vector<ge::AnyValue::DATA_TYPE> g_geAttrValueListDataType;
std::vector<std::vector<int64_t>> g_geAttrValueListListInt;
std::vector<std::vector<float, std::allocator<float>> ,std::allocator<std::vector<float, std::allocator<float> > > > g_geAttrValueListListListInt;
ge::AnyValue::ValueType g_geAttrValueType = ge::AnyValue::VT_FLOAT;

std::map<string, AnyValue> g_geAttrMap;
}

    TensorDesc::TensorDesc(void)
    {
    }

    TensorDesc::TensorDesc(TensorDesc const& desc)
    {
    }

    Shape::Shape() {}

    size_t Shape::GetDimNum() const {
        return 0;
    }

    int64_t Shape::GetDim(size_t index) const {
        return 0;
    }

    Shape TensorDesc::GetOriginShape() const {
        return {};
    }

    Format TensorDesc::GetOriginFormat() const {
        return FORMAT_ND;
    }

    GeExecutor::GeExecutor(void)
    {
    }
    Status GeExecutor::SetAllocator(void *const stream, ge::Allocator *const external_allocator){
        return MockFunctionTest::aclStubInstance().SetAllocator(stream, external_allocator);
    }

    Status GeExecutor::Initialize()
    {
        return SUCCESS;
    }

    Status GeExecutor::Finalize()
    {
        return MockFunctionTest::aclStubInstance().Finalize();
    }

    Status GeExecutor::UnloadSingleOp(const uint64_t op_id)
    {
        return SUCCESS;
    }

    Status GeExecutor::UnloadDynamicSingleOp(const uint64_t op_id)
    {
        return SUCCESS;
    }

    Status GeExecutor::RecoverAllModel(const int32_t device_id) const
    {
        return SUCCESS;
    }

    Status GeExecutor::CommandHandle(const ge::Command &command) const
    {
        return SUCCESS;
    }

    Status GeExecutor::GetDeviceIdByModelId(uint32_t model_id, uint32_t &device_id)
    {
        return SUCCESS;
    }

    ge::Status GeExecutor::SetDump(const ge::DumpConfig &dumpConfig)
    {
        return MockFunctionTest::aclStubInstance().SetDump(dumpConfig);
    }

    Status GeExecutor::ReleaseSingleOpResource(void *stream)
    {
        return SUCCESS;
    }

    Status GeExecutor::ReleaseResource()
    {
        return SUCCESS;
    }

    Status GeExecutor::GetModelDescInfo(uint32_t modelId, std::vector<TensorDesc>& inputDesc,
                                            std::vector<TensorDesc>& outputDesc, bool new_model_desc)
    {
        return MockFunctionTest::aclStubInstance().GetModelDescInfo(modelId, inputDesc, outputDesc, new_model_desc);
    }

    Status GeExecutor::GetModelDescInfoFromMem(const ModelData &model_data, ModelInOutInfo &info) const
    {
      return MockFunctionTest::aclStubInstance().GetModelDescInfoFromMem(model_data, info);
    }

    Status GeExecutor::SetDynamicAippData(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                            const std::vector<kAippDynamicBatchPara> &aippBatchPara,
                                            const kAippDynamicPara &aippParms)
    {
        return MockFunctionTest::aclStubInstance().SetDynamicAippData(model_id, dynamic_input_addr, length, aippBatchPara, aippParms);
    }

    Status GeExecutor::GetAIPPInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_params)
    {
        aipp_params.input_format = 1;
        aipp_params.related_input_rank = 0;
        aipp_params.max_src_image_size = 150528;
        return MockFunctionTest::aclStubInstance().GetAIPPInfo(model_id, index, aipp_params);
    }

    Status GeExecutor::GetAippType(uint32_t model_id, uint32_t index, ge::InputAippType &type, size_t &aippindex)
    {
        return MockFunctionTest::aclStubInstance().GetAippType(model_id, index, type, aippindex);
    }

    Status GeExecutor::GetBatchInfoSize(uint32_t model_id, size_t &shape_count)
    {
        shape_count = 1;
        return MockFunctionTest::aclStubInstance().GetBatchInfoSize(model_id, shape_count);
    }

    Status GeExecutor::GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &origOutputInfo)
    {
        origOutputInfo.format = static_cast<Format>(1);
        origOutputInfo.data_type = static_cast<DataType>(4);
        origOutputInfo.dim_num = 4;
        return MockFunctionTest::aclStubInstance().GetOrigInputInfo(model_id, index, origOutputInfo);
    }

    Status GeExecutor::GetAllAippInputOutputDims(uint32_t model_id, uint32_t index,
                                                 std::vector<InputOutputDims> &input_dims,
                                                 std::vector<InputOutputDims> &output_dims)
    {
        InputOutputDims inputDims1;
        inputDims1.dim_num = 4;
        inputDims1.dims.push_back(1);
        inputDims1.dims.push_back(224);
        inputDims1.dims.push_back(224);
        inputDims1.dims.push_back(3);
        input_dims.push_back(inputDims1);
        output_dims.push_back(inputDims1);
        return MockFunctionTest::aclStubInstance().GetAllAippInputOutputDims(model_id, index, input_dims, output_dims);
    }

    Status GeExecutor::SetDynamicBatchSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t batch_size)
    {
        return MockFunctionTest::aclStubInstance().SetDynamicBatchSize(model_id, dynamic_input_addr, length, batch_size);
    }

    Status GeExecutor::SetDynamicImageSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t image_height, uint64_t image_width)
    {
        return MockFunctionTest::aclStubInstance().SetDynamicImageSize(model_id, dynamic_input_addr, length, image_height, image_width);
    }

    Status GeExecutor::SetDynamicDims(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                      const vector<uint64_t> &dynamic_dims)
    {
      return MockFunctionTest::aclStubInstance().SetDynamicDims(model_id, dynamic_input_addr, length, dynamic_dims);
    }

    Status GeExecutor::GetCurDynamicDims(uint32_t model_id, const vector<uint64_t> &dynamic_dims,
                                         vector<uint64_t> &cur_dynamic_dims)
    {
      return MockFunctionTest::aclStubInstance().GetCurDynamicDims(model_id, dynamic_dims, cur_dynamic_dims);
    }

    int64_t TensorDesc::GetSize() const
    {
        return 1;
    }

    std::string TensorDesc::GetName() const
    {
        return "resnet50";
    }

    graphStatus TensorDesc::GetName(AscendString &name)
    {
        return MockFunctionTest::aclStubInstance().GetName(name);
    }

    AscendString::AscendString(char const *name) { }

    Format TensorDesc::GetFormat() const
    {
        return MockFunctionTest::aclStubInstance().GetFormat();
    }

    DataType TensorDesc::GetDataType() const
    {
        DataType dt = DT_FLOAT;
        return dt;
    }

    Shape TensorDesc::GetShape() const
    {
        std::vector<int64_t> vec;
        Shape shape(vec);
        return shape;
    }

    graphStatus TensorDesc::GetShapeRange(std::vector<std::pair<int64_t,int64_t>> &range) const
    {
        return MockFunctionTest::aclStubInstance().GetShapeRange(range);
    }

    const char* AscendString::GetString() const
    {
        return "resnet50";
    }

    std::vector<int64_t> Shape::GetDims() const
    {
        std::vector<int64_t> vec;
        vec.push_back(1);
        return vec;
    }

    Status GeExecutor::GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                           int32_t &dynamic_type)
    {
        return MockFunctionTest::aclStubInstance().GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
    }

    Status GeExecutor::GetCombinedDynamicDims(uint32_t model_id, vector<vector<int64_t>> &batch_info)
    {
      return ge::SUCCESS;
    }

    Status GeExecutor::GetUserDesignateShapeOrder(uint32_t model_id, vector<string> &user_designate_shape_order)
    {
        return MockFunctionTest::aclStubInstance().GetUserDesignateShapeOrder(model_id, user_designate_shape_order);
    }

    Status GeExecutor::GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type)
    {
        return MockFunctionTest::aclStubInstance().GetCurShape(model_id, batch_info, dynamic_type);
    }

    Status GeExecutor::GetModelAttr(uint32_t model_id,std::vector<std::string> &dynamic_output_shape_info)
    {
        dynamic_output_shape_info.push_back({"0:0:1,3,224,224"});
        return MockFunctionTest::aclStubInstance().GetModelAttr(model_id, dynamic_output_shape_info);
    }

    Status GeExecutor::GetOpAttr(uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                       std::string &attr_value)
    {
        return MockFunctionTest::aclStubInstance().GetOpAttr(model_id, op_name, attr_name, attr_value);
    }

    graphStatus GeExecutor::LoadDataFromFile(std::string const &path, ModelData &modelData)
    {
        return MockFunctionTest::aclStubInstance().LoadDataFromFile(path, modelData);
    }

    Status GeExecutor::LoadModelFromData(uint32_t &model_id, const ModelData &modelData,
                                   void *dev_ptr, size_t memsize, void *weight_ptr, size_t weightsize)
    {
        return MockFunctionTest::aclStubInstance().LoadModelFromData(model_id, modelData, dev_ptr, memsize, weight_ptr, weightsize);
    }

    Status GeExecutor::LoadModelFromDataWithArgs(uint32_t &model_id, const ModelData &model_data, const ModelLoadArg &load_arg)
    {
      return MockFunctionTest::aclStubInstance().LoadModelFromDataWithArgs(model_id, model_data, load_arg);
    }

    Status GeExecutor::LoadModelWithQ(uint32_t &model_id, const ge::ModelData &ge_model_data,
                   const std::vector<uint32_t> &input_queue_ids, const std::vector<uint32_t> &output_queue_ids)
    {
        return MockFunctionTest::aclStubInstance().LoadModelWithQ(model_id, ge_model_data, input_queue_ids, output_queue_ids);
    }

    Status GeExecutor::LoadModelWithQ(uint32_t &model_id, const ge::ModelData &ge_model_data,
                   const ge::ModelQueueArg &queue_arg)
    {
        return MockFunctionTest::aclStubInstance().LoadModelWithQ(model_id, ge_model_data, queue_arg);
    }

    Status GeExecutor::UnloadModel(uint32_t modelId)
    {
        return MockFunctionTest::aclStubInstance().UnloadModel(modelId);
    }

    Status GeExecutor::ExecModel(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                                const std::vector<ge::GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                                std::vector<ge::GeTensorDesc> &output_desc, bool async_mode)
    {
        return MockFunctionTest::aclStubInstance().ExecModel(model_id, stream, run_input_data, input_desc,
            run_output_data, output_desc, async_mode);
    }

    Status GeExecutor::GetMemAndWeightSize(const void *model_data, size_t model_size, size_t &mem_size, size_t &weight_size)
    {
        return MockFunctionTest::aclStubInstance().GetMemAndWeightSize(model_data, model_size, mem_size, weight_size);
    }

    Status GeExecutor::GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size)
    {
        return MockFunctionTest::aclStubInstance().GetMemAndWeightSize(path, mem_size, weight_size);
    }

    Status GeExecutor::ExecuteAsync(SingleOp *executor,
                                    const std::vector<DataBuffer> &inputs,
                                    std::vector<DataBuffer> &outputs)
    {
        return MockFunctionTest::aclStubInstance().ExecuteAsync(executor, inputs, outputs);
    }

    Status GeExecutor::ExecuteAsync(DynamicSingleOp *executor,
                        const std::vector<GeTensorDesc> &input_desc,
                        const std::vector<DataBuffer> &inputs,
                        std::vector<GeTensorDesc> &output_desc,
                        std::vector<DataBuffer> &outputs)
    {
        return MockFunctionTest::aclStubInstance().ExecuteAsync(executor, input_desc, inputs, output_desc, outputs);
    }

    Status GeExecutor::LoadSingleOpV2(const std::string &modelName,
                                      const ModelData &modelData,
                                      void *stream,
                                      SingleOp **single_op,
                                      const uint64_t model_id)
    {
        return MockFunctionTest::aclStubInstance().LoadSingleOpV2(modelName, modelData, stream, single_op, model_id);
    }

    Status GeExecutor::LoadDynamicSingleOpV2(const std::string &model_name,
                               const ge::ModelData &modelData,
                               void *stream,
                               DynamicSingleOp **single_op,
                               const uint64_t model_id)
    {
        return MockFunctionTest::aclStubInstance().LoadDynamicSingleOpV2(model_name, modelData, stream, single_op, model_id);
    }

    Status GeExecutor::GetOpDescInfo(uint32_t device_id, uint32_t stream_id,
        uint32_t task_id, OpDescInfo &op_desc_info)
    {
        op_desc_info.op_name = "cast";
        op_desc_info.input_format.push_back(FORMAT_NCHW);
        op_desc_info.input_format.push_back(FORMAT_NCHW);
        op_desc_info.output_format.push_back(FORMAT_NCHW);
        op_desc_info.output_format.push_back(FORMAT_NCHW);
        op_desc_info.input_data_type.push_back(DT_FLOAT);
        op_desc_info.input_data_type.push_back(DT_FLOAT);
        op_desc_info.output_data_type.push_back(DT_FLOAT);
        op_desc_info.output_data_type.push_back(DT_FLOAT);
        op_desc_info.input_shape.push_back({1, 1});
        op_desc_info.input_shape.push_back({1, 1});
        op_desc_info.output_shape.push_back({1, 1});
        op_desc_info.output_shape.push_back({1, 1});
        int a = 0;
        void *p = (void *)&a;
        op_desc_info.input_addrs.push_back(p);
        op_desc_info.input_addrs.push_back(p);
        op_desc_info.output_addrs.push_back(p);
        op_desc_info.output_addrs.push_back(p);
        return SUCCESS;
    }

    Model::Model(void)
    {
    }

    ProtoAttrMap &Model::MutableAttrMap()
    {
    }

    ConstProtoAttrMap &Model::GetAttrMap() const
    {
    }

    graphStatus Model::Load(const uint8_t *data, size_t len, Model &model)
    {
        return MockFunctionTest::aclStubInstance().Load(data, len, model);
    }

    AnyValue NamedAttrs::GetItem(const string &key) const
    {
        return g_geAttrValue;
    }

    ProtoAttrMap &NamedAttrs::MutableAttrMap()
    {
    }

    ConstProtoAttrMap &NamedAttrs::GetAttrMap() const
    {
    }

    AttrStore::AttrStore(AttrStore const&)
    {
    }

    AttrStore &AttrStore::operator=(const AttrStore& other)
    {
        return const_cast<AttrStore&>(other);
    }

    bool AttrUtils::GetListTensor(
        ge::AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, vector<ConstGeTensorPtr> &value)
    {
        return MockFunctionTest::aclStubInstance().GetListTensor(
            std::move(AttrUtils::ConstAttrHolderAdapter(obj)), name, value);
    }

    bool AttrUtils::GetStr(AttrUtils::ConstAttrHolderAdapter&& obj, const string& name, string& value)
    {
        return true;
    }

    bool AttrUtils::SetStr(AttrHolderAdapter &&obj, const string &name, const string &value)
    {
        return true;
    }

    bool  AttrUtils::SetListInt(AttrHolderAdapter &&obj, const string &name, const vector<int32_t> &value)
    {
        return true;
    }

    bool AttrUtils::SetBool(AttrHolderAdapter &&obj, const string &name, const bool &value)
    {
        return true;
    }

    bool AttrUtils::GetBool(ConstAttrHolderAdapter &&obj, const string &name, bool &value)
    {
        return MockFunctionTest::aclStubInstance().GetBool(
            std::move(AttrUtils::ConstAttrHolderAdapter(obj)), name, value);
    }

    bool AttrUtils::HasAttr(ConstAttrHolderAdapter &&obj, const string &name)
    {
        return MockFunctionTest::aclStubInstance().HasAttr(std::move(AttrUtils::ConstAttrHolderAdapter(obj)), name);
    }

    bool AttrUtils::SetTensor(AttrHolderAdapter &&obj, const string &name, const ConstGeTensorPtr &value)
    {
        return true;
    }

    bool AttrUtils::GetTensor(ConstAttrHolderAdapter &&obj, const string &name, ConstGeTensorPtr &value)
    {
        return true;
    }

    bool AttrUtils::GetListNamedAttrs(
        ge::AttrUtils::ConstAttrHolderAdapter &&obj, std::string const &name, vector<AnyValue::NAMED_ATTRS> &value)
    {
        return MockFunctionTest::aclStubInstance().GetListNamedAttrs(
            std::move(AttrUtils::ConstAttrHolderAdapter(obj)), name, value);
    }

    graphStatus GraphUtils::DumpGEGraphByPath(const ge::ComputeGraphPtr &graph,
        const std::string &file_path, const int64_t dump_level)
    {
        return SUCCESS;
    }

    TensorData::TensorData()
    {
    }

    TensorData::~TensorData()
    {
    }

    uint8_t* TensorData::data()
    {
        static uint8_t data[8] = {0};
        return data;
    }

    const uint8_t* TensorData::data() const
    {
        static uint8_t data[8] = {0};
        return data;
    }

    size_t TensorData::size() const
    {
        return 8U;
    }

    std::uint8_t *TensorData::GetData()
    {
        return nullptr;
    }

    const std::uint8_t *TensorData::GetData() const
    {
        return nullptr;
    }

    const TensorData &GeTensor::GetData() const
    {
        static TensorData td;
        return td;
    }

    std::size_t TensorData::GetSize() const
    {
        return 0;
    }

    Buffer::Buffer()
    {
    }

    std::uint8_t *Buffer::GetData()
    {
        return nullptr;
    }

    const std::uint8_t *Buffer::GetData() const
    {
        return nullptr;
    }

    std::size_t Buffer::GetSize() const
    {
        return 0;
    }

    graphStatus AttrHolder::DelAttr(std::string const&)
    {
        return 0;
    }

    std::vector<int64_t> GeShape::GetDims() const
    {
        vector<int64_t> vec;
        return vec;
    }

    size_t GeShape::GetDimNum() const
    {
        return 0;
    }

    void GeShape::SetDimNum(size_t dim_num)
    {
       return;
    }

    graphStatus GeShape::SetDim(size_t idx, int64_t value)
    {
        return GRAPH_SUCCESS;
    }

    GeShape &GeShape::operator=(const GeShape &other)
    {
        return const_cast<GeShape&>(other);
    }

    GeShape &GeShape::operator=(GeShape &&other)
    {
        return other;
    }

    const std::map<string, AnyValue> AttrHolder::GetAllAttrs() const
    {
        return MockFunctionTest::aclStubInstance().GetAllAttrs();
        
    }

    GeTensorDesc::GeTensorDesc()
    {
    }

    GeTensorDesc::~GeTensorDesc()
    {
    }

    GeTensorDesc::GeTensorDesc(GeTensorDesc&&)
    {
    }

    GeTensorDesc::GeTensorDesc(GeTensorDesc const&)
    {
    }

    GeTensorDesc::GeTensorDesc(const GeShape &shape, Format format, DataType dt)
    {
    }

    ProtoAttrMap &GeTensorDesc::MutableAttrMap()
    {
    }

    ConstProtoAttrMap &GeTensorDesc::GetAttrMap() const
    {
    }

    void GeTensorDesc::SetOriginFormat(Format originFormat)
    {
    }

    void GeTensorDesc::SetFormat(Format format)
    {
        return;
    }

    Format GeTensorDesc::GetFormat() const
    {
        return FORMAT_NCHW;
    }

    Format GeTensorDesc::GetOriginFormat() const
    {
        return FORMAT_NCHW;
    }

    GeShape& GeTensorDesc::MutableShape()
    {
        static GeShape shape;
        return shape;
    }

    DataType GeTensorDesc::GetDataType() const
    {
        return DT_FLOAT;
    }

    void GeTensorDesc::SetShape(const GeShape &shape)
    {
        return;
    }

    void GeTensorDesc::SetShape(GeShape &&shape) 
    {
        return;
    }

    const GeShape &GeTensorDesc::GetShape() const
    {
        static GeShape shape({1, 2, 3, 4});
        return shape;
    }

    const GeShape &GeTensorDesc::GetOriginShape() const
    {
        static GeShape shape({1, 2, 3, 4});
        return shape;
    }

    void GeTensorDesc::SetOriginShape(const GeShape &originShape)
    {
        return;
    }

    graphStatus GeTensorDesc::SetShapeRange(const std::vector<std::pair<int64_t,int64_t>> &range)
    {
        return MockFunctionTest::aclStubInstance().SetShapeRange(range);
    }

    graphStatus GeTensorDesc::SetValueRange(const std::vector<std::pair<int64_t, int64_t>> &range)
    {
        return SUCCESS;
    }

    void GeTensorDesc::SetDataType(ge::DataType dt)
    {
        return;
    }

    void GeTensorDesc::SetOriginDataType(DataType origin_data_type)
    {
        return;
    }

    graphStatus GeTensorDesc::GetShapeRange(std::vector<std::pair<int64_t,int64_t>> &range) const
    {
        range.push_back(std::make_pair(1, 16));
        range.push_back(std::make_pair(1, 16));
        range.push_back(std::make_pair(1, 16));
        range.push_back(std::make_pair(1, 16));
        return GRAPH_SUCCESS;
    }

    GeTensor::GeTensor()
    {
    }

    GeTensor::~GeTensor()
    {
    }

    GeTensor::GeTensor(GeTensorDesc const&)
    {
    }

    GeTensor::GeTensor(GeTensor const&)
    {
    }

    GeTensor::GeTensor(GeTensor &&other) noexcept
    {
    }

    TensorData &GeTensor::MutableData()
    {
        TensorData tensorData;
        return tensorData;
    }

    GeTensor::GeTensor(const GeTensorDesc &tensorDesc, const uint8_t *data, size_t size)
    {
    }

    const GeTensorDesc &GeTensor::GetTensorDesc() const
    {
        GeTensorDesc tenosrDesc;
        return tenosrDesc;
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<bool>() {
        return reinterpret_cast<TypeId>(1);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::string>() {
        return reinterpret_cast<TypeId>(2);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<float>() {
        return reinterpret_cast<TypeId>(3);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<int64_t>() {
        return reinterpret_cast<TypeId>(4);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<GeTensorDesc>() {
        return reinterpret_cast<TypeId>(5);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<GeTensor>() {
        return reinterpret_cast<TypeId>(6);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<Buffer>() {
        return reinterpret_cast<TypeId>(7);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<proto::GraphDef>() {
        return reinterpret_cast<TypeId>(8);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<NamedAttrs>() {
        return reinterpret_cast<TypeId>(9);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::vector<int64_t>>>() {
        return reinterpret_cast<TypeId>(10);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<DataType>() {
    return reinterpret_cast<TypeId>(11);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::vector<float>>>() {
        return reinterpret_cast<TypeId>(12);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::string>>() {
        return reinterpret_cast<TypeId>(13);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<float>>() {
        return reinterpret_cast<TypeId>(14);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<bool>>() {
        return reinterpret_cast<TypeId>(15);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<int64_t>>() {
        return reinterpret_cast<TypeId>(16);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<GeTensorDesc>>() {
        return reinterpret_cast<TypeId>(17);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<GeTensor>>() {
        return reinterpret_cast<TypeId>(18);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<Buffer>>() {
        return reinterpret_cast<TypeId>(19);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<proto::GraphDef>>() {
        return reinterpret_cast<TypeId>(20);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<NamedAttrs>>() {
        return reinterpret_cast<TypeId>(21);
    }

    template<>
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<DataType>>() {
        return reinterpret_cast<TypeId>(22);
    }

    void AnyValue::Swap(AnyValue &other) noexcept {
        AnyValue tmp;
        if (!other.IsEmpty()) {
            other.operate_(OperateType::kOpMove, &other, &tmp);
        }

        other.Clear();
        if (!IsEmpty()) {
            operate_(OperateType::kOpMove, this, &other);
        }

        Clear();
        if (!tmp.IsEmpty()) {
            tmp.operate_(OperateType::kOpMove, &tmp, this);
        }
        }

        AnyValue::AnyValue(AnyValue &&other) noexcept {
        if (!other.IsEmpty()) {
            other.operate_(OperateType::kOpMove, &other, this);
        }
        }
        AnyValue &AnyValue::operator=(AnyValue &&other) noexcept {
        if (&other == this) {
            return *this;
        }
        Clear();
        if (!other.IsEmpty()) {
            other.operate_(OperateType::kOpMove, &other, this);
        }
        return *this;
        }
        AnyValue &AnyValue::operator=(const AnyValue &other) {
        if (&other == this) {
            return *this;
        }
        Clear();
        if (!other.IsEmpty()) {
            other.operate_(OperateType::kOpClone, &other, this);
        }
        return *this;
        }
        TypeId AnyValue::GetValueTypeId() const noexcept {
        TypeId vt{kInvalidTypeId};
        if (!IsEmpty()) {
            operate_(OperateType::kGetTypeId, this, &vt);
        }
        return vt;
        }
        AnyValue::ValueType AnyValue::GetValueType() const noexcept {
        auto vt = GetValueTypeId();
        auto iter = type_ids_to_value_type.find(vt);
        if (iter == type_ids_to_value_type.end()) {
            return AnyValue::VT_NONE;
        }
        return iter->second;
        }
        AnyValue AnyValue::Copy() const {
        AnyValue av(*this);
        return av;
        }
        const void *AnyValue::GetAddr() const {
        void *addr = nullptr;
        operate_(OperateType::kOpGetAddr, this, &addr);
        return addr;
        }

    graphStatus OpDesc::AddInputDesc(const string &name, const GeTensorDesc &input_desc)
    {
        return GRAPH_SUCCESS;
    }

    graphStatus OpDesc::AddOutputDesc(const string &name, const GeTensorDesc &output_desc)
    {
        return GRAPH_SUCCESS;
    }

    GeTensorDescPtr OpDesc::MutableInputDesc(const uint32_t index) const
    {
        std::shared_ptr<GeTensorDesc> in_desc = std::make_shared<GeTensorDesc>();
        return in_desc;
    }

    OpDesc::OpDesc()
    {
    }

    OpDesc::~OpDesc()
    {
    }

    std::string OpDesc::GetName() const
    {
        return "OpName";
    }

    OpDesc::OpDesc(std::string const&, std::string const&)
    {
    }

    std::string OpDesc::GetInputNameByIndex(uint32_t index) const
    {
        return "";
    }

    ProtoAttrMap &OpDesc::MutableAttrMap()
    {
    }

    ConstProtoAttrMap &OpDesc::GetAttrMap() const
    {
    }


    void TensorUtils::SetRealDimCnt(GeTensorDesc& tensorDesc, uint32_t cnt)
    {
    }

    void TensorUtils::SetInputTensor(GeTensorDesc& tensorDesc, bool flag)
    {
    }

    void TensorUtils::SetOutputTensor(GeTensorDesc& tensorDesc, bool flag)
    {
    }


    graphStatus OpDesc::AddInputDesc(const GeTensorDesc& input_desc)
    {
        return GRAPH_SUCCESS;
    }

    graphStatus OpDesc::AddOutputDesc(const GeTensorDesc& output_desc)
    {
        return GRAPH_SUCCESS;
    }

    GeShape::GeShape()
    {
    }

    GeShape::~GeShape()
    {
    }

    GeShape::GeShape(GeShape const&)
    {
    }

    graphStatus AttrHolder::SetAttr(const string& name, const AnyValue& value)
    {
        return GRAPH_SUCCESS;
    }

    GeShape::GeShape(std::vector<long>)
    {
    }

    Status GEInitialize(const std::map<AscendString, AscendString>& options)
    {
        return MockFunctionTest::aclStubInstance().GEInitialize(options);
    }

    bool AscendString::operator<(const AscendString& d) const
    {
        return true;
    }

    bool AscendString::operator>(const AscendString& d) const
    {
        return true;
    }

    bool AscendString::operator<=(const AscendString& d) const
    {
        return true;
    }

    bool AscendString::operator>=(const AscendString& d) const
    {
        return true;
    }

    bool AscendString::operator==(const AscendString& d) const
    {
        return true;
    }

    bool AscendString::operator!=(const AscendString& d) const
    {
        return true;
    }

    Status GEFinalize()
    {
        return MockFunctionTest::aclStubInstance().GEFinalize();
    }

    Status GeGenerator::Initialize(const std::map<std::string, std::string> &options)
    {
        return MockFunctionTest::aclStubInstance().Initialize(options);
    }

    Status GeGenerator::Initialize(const std::map<std::string, std::string> &options, OmgContext &omgContext)
    {
        return MockFunctionTest::aclStubInstance().Initialize(options, omgContext);
    }

    Status GeGenerator::BuildSingleOpModel(ge::OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                           const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                           int32_t compile_flag, ModelBufferData &model_buff)
    {
        return MockFunctionTest::aclStubInstance().BuildSingleOpModel(op_desc, inputs, outputs, engine_type,
                                                                      compile_flag, model_buff);
    }

    Status GeGenerator::BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                            int32_t compile_flag, ModelBufferData &model_buff,
                            GraphStage graph_stage, ComputeGraphPtr &compute_graph)
    {
        return MockFunctionTest::aclStubInstance().BuildSingleOpModel(op_desc, inputs, outputs, engine_type,
                                                                      compile_flag, model_buff,
                                                                      graph_stage, compute_graph);
    }

    Status GeGenerator::Finalize()
    {
        return MockFunctionTest::aclStubInstance().Ge_Generator_Finalize();
    }

    Shape::Shape(const std::vector<int64_t>& dims)
    {
    }

    TensorDesc::TensorDesc(Shape shape, Format format, DataType dt)
    {
    }

    Status GeFormatUtil::TransShape(const TensorDesc &src_desc,
                                    Format dst_format,
                                    std::vector<int64_t> &dst_shape)
    {
        return MockFunctionTest::aclStubInstance().TransShape(src_desc, dst_format, dst_shape);
    }

    bool AttrUtils::GetInt(ge::AttrUtils::ConstAttrHolderAdapter&& obj, const std::string &name, int64_t &value)
    {
        return true;
    }

    bool AttrUtils::GetInt(ge::AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &name, int32_t &value)
    {
        return MockFunctionTest::aclStubInstance().GetInt(
            std::move(AttrUtils::ConstAttrHolderAdapter(obj)), name, value);
    }

    bool AttrUtils::SetInt(ge::AttrUtils::AttrHolderAdapter &&obj, const string &name, const int64_t &value)
    {
       return true;
    }

    bool AttrUtils::SetListInt(ge::AttrUtils::AttrHolderAdapter &&obj, const string &name, const vector<int64_t> &value)
    {
       return true;
    }

    bool AttrUtils::GetListInt(ge::AttrUtils::ConstAttrHolderAdapter &&obj, std::string const&, std::vector<int64_t> &value)
    {
        return true;
    }

} // namespace ge

namespace ge {
    Status OmFileLoadHelper::Init(uint8_t *model_data, const uint32_t model_data_size)
    {
        return MockFunctionTest::aclStubInstance().Init(model_data, model_data_size);
    }

    Status OmFileLoadHelper::Init(uint8_t *model_data, const uint32_t model_data_size, const uint32_t model_num)
    {
        return SUCCESS;
    }

    Status OmFileLoadHelper::Init(uint8_t *const model_data,
                                  const uint64_t model_data_size,
                                  const ModelFileHeader *file_header)
    {
        return MockFunctionTest::aclStubInstance().Init(model_data, model_data_size);
    }

    Status OmFileLoadHelper::Init(uint8_t *const model_data,
                                  const uint64_t model_data_size,
                                  const uint32_t model_num,
                                  const ModelFileHeader *file_header)
    {
        return SUCCESS;
    }

    Status OmFileLoadHelper::GetModelPartition(const ModelPartitionType type, ModelPartition &partition)
    {
        return MockFunctionTest::aclStubInstance().GetModelPartition(type, partition);
    }

    Status OmFileLoadHelper::GetModelPartition(const ModelPartitionType type,
        ModelPartition &partition, const size_t model_index) const
    {
        return SUCCESS;
    }

    Status OmFileLoadHelper::CheckModelCompatibility(const Model &model) const
    {
       return SUCCESS;
    }

    bool ReadBytesFromBinaryFile(char const *file_name, char **buffer, int &length)
    {
        return MockFunctionTest::aclStubInstance().ReadBytesFromBinaryFile(file_name, buffer, length);
    }

    std::string RealPath(const char *path)
    {
        return MockFunctionTest::aclStubInstance().RealPath(path);
    }

    TensorDesc Operator::GetOutputDesc(uint32_t index) const
    {
        std::vector<int64_t> vec;
        Shape shape(vec);
        Format format = FORMAT_NCHW;
        DataType dt = DT_FLOAT;
        TensorDesc tensorDesc(shape, format, dt);
        return tensorDesc;
    }

    Operator &Operator::SetAttr(const string &name, const Tensor &attr_value)
    {

    }

    Operator &Operator::SetAttr(const char *name, const Tensor &attr_value) {
        return *this;
    }

    void Operator::BreakConnect() const {
        return;
    }

    graphStatus Operator::InferShapeAndType()
    {
        return 0;
    }

    Operator &Operator::SetInput(const string &dst_name, const Operator &src_oprt, const string &name)
    {
        return *this;
    }

    Operator &Operator::SetInput(const char *dst_name, const Operator &src_oprt, const char *name){
        return *this;
    }

    OpsProtoManager *OpsProtoManager::Instance() {
        static OpsProtoManager instance;
        return &instance;
    }

    bool OpsProtoManager::Initialize(const std::map<std::string, std::string> &options)
    {
        return MockFunctionTest::aclStubInstance().OpsProtoManager_Initialize(options);
    }

    OpsProtoManager::~OpsProtoManager() {}

    Operator OperatorFactory::CreateOperator(const std::string &operator_name, const std::string &operator_type)
    {
        Operator op;
        return op;
    }

    Operator OperatorFactory::CreateOperator(const char *operator_name, const char *operator_type)
    {
        Operator op;
        return op;
    }

    graphStatus OperatorFactory::GetOpsTypeList(std::vector<std::string> &all_ops)
    {
        return 0;
    }

    graphStatus OperatorFactory::GetOpsTypeList(std::vector<ge::AscendString> &all_ops)
    {
        return MockFunctionTest::aclStubInstance().GetOpsTypeList(all_ops);;
    }

    OpDescPtr OpDescUtils::GetOpDescFromOperator(const Operator& oprt)
    {
        return std::make_shared<ge::OpDesc>("default_op_desc", "default_op_desc");
    }

    Operator OpDescUtils::CreateOperatorFromOpDesc(OpDescPtr op_desc)
    {
        Operator op;
        return op;
    }

    Tensor::Tensor(const TensorDesc &tensorDesc, const uint8_t *data, size_t size)
    {

    }

    std::map<string, uint32_t> OpDesc::GetAllInputName() const
    {
        std::map<string, uint32_t> mapStub;
        return mapStub;
    }

    GEThreadLocalContext &GetThreadLocalContext() { return threadContext; }

    void GEThreadLocalContext::SetGlobalOption(map<string, string> options_map)
    {
        ge_thread_global_options = options_map;
        return;
    }

    void GEThreadLocalContext::SetSessionOption(map<string, string> options_map) {
        return;
    }

    void GEThreadLocalContext::SetGraphOption(map<string, string> options_map) {
        return;
    }

    std::map<std::string, std::string> GEThreadLocalContext::GetAllGraphOptions() const {
        return ge_thread_global_options;
    }

    GEContext &GetContext() {
        return geContext;
    }

    void GEContext::SetStreamSyncTimeout(const int32_t timeout) {
        ge_context_stream_sync_timeout = timeout;
    }

    void GEContext::SetEventSyncTimeout(const int32_t timeout) {
        ge_context_event_sync_timeout = timeout;
    }

    graphStatus GEContext::SetOptionNameMap(const std::string &option_name_map_json) {
        nlohmann::json option_json;
        try {
            option_json = nlohmann::json::parse(option_name_map_json);
        } catch (nlohmann::json::parse_error&) {
            return ge::GRAPH_FAILED;
        }
        for (auto iter : option_json.items()) {
            if (iter.key().empty()) {
                return ge::GRAPH_FAILED;
            }
            if (static_cast<std::string>(iter.value()).empty()) {
                return ge::GRAPH_FAILED;
            }
            ge_context_option_name_map.insert({iter.key(), static_cast<std::string>(iter.value())});
        }
        return ge::GRAPH_SUCCESS;
    }

    const std::string & GEContext::GetReadableName(const std::string &key) {
        auto iter = ge_context_option_name_map.find(key);
        if (iter != ge_context_option_name_map.end()) {
            return iter->second;
        }
        return key;
    }

    int32_t GEContext::StreamSyncTimeout() const {
        return ge_context_stream_sync_timeout;
    }

    int32_t GEContext::EventSyncTimeout() const {
        return ge_context_event_sync_timeout;
    }

    bool TypeUtilsInner::IsInternalFormat(ge::Format format) {
        if (format == FORMAT_NCHW) {
            return false;
        }
        return true;
    }

const uint32_t MODEL_FILE_MAGIC_NUM = 0x444F4D49;
const uint32_t MODEL_FILE_HEAD_LEN = 256;
const uint32_t MODEL_VERSION = 0x10000000;

}// namespace ge

ge::Status RegProfReporterCallback(MsprofReporterCallback func)
{
    return 0;
}

std::unique_ptr<const char_t[]> aclStub::GetErrMgrErrorMessage()
{
    const char *str = "default";
    std::unique_ptr<const char[]> errMsg(new char[std::strlen(str) + 1]);
    std::strcpy(const_cast<char*>(errMsg.get()), str);
    return errMsg;
}

int error_message::ErrMgrInit(error_message::ErrorMessageMode error_mode)
{
    (void)error_mode;
    return MockFunctionTest::aclStubInstance().Init();
}

std::unique_ptr<const char_t[]> error_message::GetErrMgrErrorMessage()
{
    return MockFunctionTest::aclStubInstance().GetErrMgrErrorMessage();
}

int32_t error_message::ReportPredefinedErrMsg(const char *error_code) {
    return 0;
}

int32_t error_message::ReportPredefinedErrMsg(const char *error_code, const std::vector<const char *> &key,
                               const std::vector<const char *> &value)
{
    return 0;
}

int32_t error_message::ReportInnerErrMsg(const char *file_name, const char *func, uint32_t line, const char *error_code,
                          const char *format, ...)
{
    return 0;
}


namespace ge {
    namespace profiling {
        ProfilingContext &ProfilingContext::GetInstance()
        {
            static ProfilingContext pc;
            return pc;
        }
        bool ProfilingContext::IsDumpToStdEnabled()
        {
            return false;
        }
        int64_t ProfilingContext::RegisterString(const std::string &str)
        {
            return -1;
        }
        ProfilingContext::ProfilingContext() = default;
        ProfilingContext::~ProfilingContext() = default;
        void Profiler::RecordCurrentThread(int64_t element, int64_t event, EventType et,
                                           std::chrono::time_point<std::chrono::system_clock> time_point)
        {
            return;
        }
        Profiler::~Profiler() = default;
    }
}

namespace ge {
    StreamMngFuncRegistry &StreamMngFuncRegistry::GetInstance()
    {
        static StreamMngFuncRegistry registry;
        return registry;
    }
    void StreamMngFuncRegistry::Register(const StreamMngFuncType func_type, const StreamMngFunc manage_func)
    {
    }
    Status StreamMngFuncRegistry::TryCallStreamMngFunc(
        const StreamMngFuncType func_type, MngActionType action_type, MngResourceHandle handle)
    {
        return SUCCESS;
    }
    StreamMngFunc StreamMngFuncRegistry::LookUpStreamMngFunc(const StreamMngFuncType func_type)
    {
        return nullptr;
    }
    StreamMngFuncRegister::StreamMngFuncRegister(const StreamMngFuncType, const StreamMngFunc func)
    {
    }

} // namespace ge

namespace gert {
    StreamAllocator::StreamAllocator(int32_t priority, uint32_t flags) {
        (void)priority;
        (void)flags;
    }
    StreamAllocator::~StreamAllocator() {}
    EventAllocator::~EventAllocator() {}
    NotifyAllocator::~NotifyAllocator() {}
    StreamExecutor::~StreamExecutor() {}
    StreamExecutor::StreamExecutor(ModelV2ExecutorBuilder *builder) {
        builder_ = builder;
    }
    ModelV2Executor *StreamExecutor::CreateAndLoad(rtStream_t stream, const gert::ModelExecuteArg &arg)
    {
        return MockFunctionTest::aclStubInstance().CreateAndLoad(stream, arg);
    }
    ge::graphStatus StreamExecutor::Erase(rtStream_t stream)
    {
        return MockFunctionTest::aclStubInstance().Erase(stream);
    }
    ge::graphStatus LoadDataFromFile(const char *path, ge::ModelData &model_data) {
        return MockFunctionTest::aclStubInstance().LoadDataFromFileV2(path, model_data);
    }
    std::unique_ptr<gert::ModelV2Executor>
    LoadExecutorFromModelDataWithMem(const ge::ModelData &model_data, ge::graphStatus &error_code,
                                     const void *weight_ptr, const size_t weight_size) {
        return MockFunctionTest::aclStubInstance().LoadExecutorFromModelDataWithMem(model_data, error_code,
                                                                                    weight_ptr, weight_size);
    }
    std::unique_ptr<ModelV2Executor> LoadExecutorFromFile(const char *file_path, ge::graphStatus &error_code)
    {
        return MockFunctionTest::aclStubInstance().LoadExecutorFromFile(file_path, error_code);
    }
    ge::graphStatus ModelV2Executor::Load()
    {
        return MockFunctionTest::aclStubInstance().Load();
    }
    ge::graphStatus ModelV2Executor::Load(const gert::ModelExecuteArg &arg)
    {
        return MockFunctionTest::aclStubInstance().Load(arg);
    }
    ge::graphStatus ModelV2Executor::Load(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg)
    {
        return MockFunctionTest::aclStubInstance().Load(arg, load_arg);
    }
    std::unique_ptr<ge::Allocator> AllocatorFactory::Create(const gert::TensorPlacement &placement) {
        return MockFunctionTest::aclStubInstance().Create(placement);
    }
    ge::Allocator *gert::Allocators::GetAllocator(const gert::TensorPlacement &placement,
                                                         const size_t &usage) {
        return nullptr;
    }
    std::unique_ptr<ge::Allocator> CreateExternalAllocator(const ge::AllocatorDesc * const allocatorDesc) {
        return MockFunctionTest::aclStubInstance().Create(gert::kOnDeviceHbm);
    }

    ge::Status Allocators::SetAllocator(const gert::TensorPlacement &placement, const size_t &usage,
                             std::shared_ptr<ge::Allocator> &allocator) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus ModelV2Executor::Execute(const gert::ModelExecuteArg &arg,
                          Tensor **inputs, size_t input_num,
                          Tensor **outputs, size_t output_num)
    {
        return MockFunctionTest::aclStubInstance().Execute(arg, inputs, input_num, outputs, output_num);
    }
    ge::graphStatus ModelV2Executor::ExecuteSync(Tensor **inputs, size_t input_num,
                              Tensor **outputs, size_t output_num)
    {
        return MockFunctionTest::aclStubInstance().ExecuteSync(inputs, input_num, outputs, output_num);
    }
    ge::graphStatus ModelV2Executor::UnLoad()
    {
        return MockFunctionTest::aclStubInstance().UnLoad();
    }

    TopologicalResourceGuard::~TopologicalResourceGuard() {
    }
    ModelV2Executor::ModelV2Executor() {
    }

    std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data, ge::graphStatus &error_code)
    {
        return MockFunctionTest::aclStubInstance().LoadExecutorFromModelData(model_data, error_code);
    }

    std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                               const LoadExecutorArgs &args,
                                                               ge::graphStatus &error_code)
    {
        return MockFunctionTest::aclStubInstance().LoadExecutorFromModelData(model_data, args, error_code);
    }

    std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromModelDataWithRtSession(const ge::ModelData &model_data,
                                                                                  gert::RtSession *const rt_session,
                                                                                  ge::graphStatus &error_code)
    {
        return MockFunctionTest::aclStubInstance().LoadExecutorFromModelDataWithRtSession(model_data, rt_session, error_code);
    }
    std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data, const void *weight_ptr,
                                                                    const size_t weight_size, ge::graphStatus &error_code)
    {
        return MockFunctionTest::aclStubInstance().LoadStreamExecutorFromModelData(model_data, weight_ptr, weight_size, error_code);
    }
    std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                          const gert::LoweringOption &optimize_option, ge::graphStatus &error_code)
    {
        return MockFunctionTest::aclStubInstance().LoadStreamExecutorFromModelData(model_data, optimize_option, error_code);
    }
    ge::graphStatus IsDynamicModel(const char *file_path, bool &is_dynamic_model)
    {
        return MockFunctionTest::aclStubInstance().IsDynamicModel(file_path, is_dynamic_model);
    }
    const ModelDesc &ModelV2Executor::GetModelDesc() const
    {
        static ModelDesc desc;
        return desc;
    }
    ge::Status ModelV2Executor::GetAippInfo(const uint32_t index, ge::AippConfigInfo &aipp_info) const {
        return MockFunctionTest::aclStubInstance().GetAippInfo(index, aipp_info);
    }

    ge::Status ModelV2Executor::GetAippType(const uint32_t index, ge::InputAippType &aipp_type, size_t &aipp_index) const {
        return MockFunctionTest::aclStubInstance().GetAippType(index, aipp_type, aipp_index);
    }

    ge::Status ModelV2Executor::GetOriginAippInputInfo(const uint32_t index, ge::OriginInputInfo &orig_input_info) const
    {
        return MockFunctionTest::aclStubInstance().GetOriginAippInputInfo(index, orig_input_info);
    }

    ge::Status ModelV2Executor::GetAllAippInputOutputDims(const uint32_t index, std::vector<ge::InputOutputDims> &input_dims,
        std::vector<ge::InputOutputDims> &output_dims) const
    {
        return MockFunctionTest::aclStubInstance().GetAllAippInputOutputDims(index, input_dims, output_dims);
    }

    ge::Status InitAipp(const ge::ComputeGraphPtr &root_graph) {
        return ge::SUCCESS;
    }

    const ModelIoDesc *ModelDesc::GetInputDesc(size_t index) const
    {
        static ModelIoDesc desc;
        return &desc;
    }
    const ModelIoDesc *ModelDesc::GetAllInputsDesc(size_t &input_num) const
    {
        static ModelIoDesc inputs;
        input_num = 1;
        return &inputs;
    }

    const ModelIoDesc *ModelDesc::GetOutputDesc(size_t index) const
    {
        static ModelIoDesc desc;
        return &desc;
    }
    
    const ModelIoDesc *ModelDesc::GetAllOutputsDesc(size_t &output_num) const
    {
        static ModelIoDesc outputs;
        output_num = 1;
        return &outputs;
    }

    ge::graphStatus ModelDesc::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus ModelDesc::GetUserDesignateShapeOrder(std::vector<std::string> &user_designate_shape_order) const
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus ModelDesc::GetModelAttrs(std::vector<std::string> &dynamic_output_shape_info) const
    {
        return ge::GRAPH_SUCCESS;
    }

    size_t ModelDesc::GetInputNum() const
    {
        return 1;
    }
    size_t ModelDesc::GetOutputNum() const
    {
        return 1;
    }

    int64_t ModelIoDesc::GetSize() const
    {
        return 0;
    }
    ge::Format ModelIoDesc::GetOriginFormat() const
    {
        return ge::FORMAT_NCHW;
    }
    const Shape &ModelIoDesc::GetOriginShape() const
    {
        return shape_.GetOriginShape();
    }
    const Shape &ModelIoDesc::GetStorageShape() const
    {
        return shape_.GetStorageShape();
    }

    const char *ModelIoDesc::GetName() const
    {
        return "";
    }

    int32_t ModelIoDesc::GetDataType() const
    {
        return 0;
    }

    const ShapeRange &ModelIoDesc::GetOriginShapeRange() const
    {
        return origin_shape_range_;
    }

    std::vector<std::pair<int64_t, int64_t>> ModelIoDesc::GetOriginShapeRangeVector() const {
        return std::vector<std::pair<int64_t, int64_t>>();
    }

    std::vector<std::pair<int64_t, int64_t>> ModelIoDesc::GetStorageShapeRangeVector() const {
        return std::vector<std::pair<int64_t, int64_t>>();
    }

    bool ModelIoDesc::IsOriginShapeInRange(const Shape &shape) const {
        return MockFunctionTest::aclStubInstance().IsOriginShapeInRange(shape);
    }

    const Shape &ModelIoDesc::GetAippShape() const {
        return aipp_shape_;
    }

    const Shape &ShapeRange::GetMin() const
    {
        return min_;
    }
    const Shape &ShapeRange::GetMax() const
    {
        return max_;
    }

    // gert profiling
    void GlobalProfiler::Dump(std::ostream &out_stream, std::vector<std::string> &idx_to_str) const {
      return;
    }

    GlobalProfilingWrapper::GlobalProfilingWrapper() {

    }
    void GlobalProfilingWrapper::OnGlobalProfilingSwitch(void *ins, uint64_t enable_flags) {
      return;
    }

    void GlobalProfilingWrapper::Init(uint64_t enable_flags) {
      return;
    }

    ge::Status GlobalProfilingWrapper::ProfileStepTrace(const uint64_t index_id, const uint32_t model_id, const uint16_t tag_id,
                             const rtStream_t stream) {
        return 0;
    }

    ge::Status GlobalProfilingWrapper::ReportEvent(const uint64_t item_id, const uint32_t request_id, GeProfInfoType type,
                                                   MsprofEvent &prof_single_event) {
        return 0;
    }

    void GlobalProfilingWrapper::RegisterBuiltInString() {
    }

    uint64_t GlobalProfilingWrapper::RegisterString(const std::string &name) {
        return 0;
    }

    void GlobalProfilingWrapper::SetModelIdStepId(const uint32_t model_id, const uint32_t step_id) {
    }

    void ExecutorSubscribersScheduler::OnExecuteEvent(SubExeGraphType sub_exe_graph_type,
                                                      const ExecutorSubscribersScheduler *ins,
                                                      ExecutorEvent event, const void *node, KernelStatus result)
    {
    }

    uint32_t ModelV2Executor::GetIterationNum() const {
        return 0;
    }

    ge::graphStatus Om2ModelExecutor::Load(ge::ModelData &model_data) const {
        (void)model_data;
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus Om2ModelExecutor::Run(std::vector<gert::Tensor *> &inputs,
                                          std::vector<gert::Tensor *> &outputs) const {
      (void)inputs;
      (void)outputs;
      return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus Om2ModelExecutor::RunAsync(void *stream, std::vector<gert::Tensor *> &inputs,
                                               std::vector<gert::Tensor *> &outputs) const {
        (void)inputs;
        (void)outputs;
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus Om2ModelExecutor::GetModelDescInfo(std::vector<ge::TensorDesc> &input_desc,
                                                       std::vector<ge::TensorDesc> &output_desc,
                                                       bool new_model_desc) const {

      return MockFunctionTest::aclStubInstance().GetModelDescInfo(input_desc, output_desc, new_model_desc);
    }
    ge::graphStatus Om2ModelExecutor::GetModelAttrs(std::vector<std::string> &dynamic_output_shape) const {
        (void)dynamic_output_shape;
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus Om2ModelExecutor::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &dynamic_batch_info,
                                                          int32_t &dynamic_type) const {
        (void)dynamic_batch_info;
        (void)dynamic_type;
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus Om2ModelExecutor::GetUserDesignateShapeOrder(
        std::vector<std::string> &user_designate_shape_order) const {
      (void)user_designate_shape_order;
      return ge::GRAPH_SUCCESS;
    }

    class Om2ModelExecutor::Impl {
      public:
        int dummy;
    };
    Om2ModelExecutor::Om2ModelExecutor() {}
    Om2ModelExecutor::~Om2ModelExecutor() {}

    ge::Status LoadOm2DataFromFile(const std::string &model_path, ge::ModelData &model_data) {
        return MockFunctionTest::aclStubInstance().LoadOm2DataFromFile(model_path, model_data);
    }
    std::unique_ptr<gert::Om2ModelExecutor> LoadOm2ExecutorFromData(ge::ModelData &model_data,
                                                                    ge::Status &error_code) {
        return MockFunctionTest::aclStubInstance().LoadOm2ExecutorFromData(model_data, error_code);
    }
    ge::Status IsOm2Model(const void *data, size_t size, bool &is_support) {
        return MockFunctionTest::aclStubInstance().IsOm2Model(data, size, is_support);
    }
    ge::Status IsOm2Model(const char *file_path, bool &is_support) {
        return MockFunctionTest::aclStubInstance().IsOm2Model(file_path, is_support);
    }

}

namespace error_message {
    int FormatErrorMessage(char *str_dst, size_t dst_max, const char *format, ...)
    {
        return 1;
    }

    std::string TrimPath(const std::string &str) {
        return "";
    }

    void ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line,
                                         const std::string error_code, const char_t *format, ...) {
        return;
    }
}

namespace fe {
    PlatformInfoManager::PlatformInfoManager() : init_flag_(false) {}

    PlatformInfoManager::~PlatformInfoManager() {}

    PlatformInfoManager &PlatformInfoManager::GeInstance() {
        static PlatformInfoManager ge_platform_info;
        return ge_platform_info;
    }

    uint32_t PlatformInfoManager::InitRuntimePlatformInfos(const std::string &SoCVersion) {
        return MockFunctionTest::aclStubInstance().InitRuntimePlatformInfos(SoCVersion);
    }

    uint32_t PlatformInfoManager::GetRuntimePlatformInfosByDevice(const uint32_t &device_id,
                                                                  PlatFormInfos &platform_infos,
                                                                  bool need_deep_copy) {
        (void) need_deep_copy;
        return MockFunctionTest::aclStubInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
    }

    uint32_t PlatformInfoManager::UpdateRuntimePlatformInfosByDevice(const uint32_t &device_id,
                                                                     PlatFormInfos &platform_infos) {                                                      
        return MockFunctionTest::aclStubInstance().UpdateRuntimePlatformInfosByDevice(device_id, platform_infos);
    }

    uint32_t fe::PlatformInfoManager::InitializePlatformInfo()
    {
        return MockFunctionTest::aclStubInstance().InitializePlatformInfo();
    }

    uint32_t fe::PlatformInfoManager::GetPlatformInfos(
        const std::string SoCVersion, fe::PlatFormInfos &platformInfo, fe::OptionalInfos &optionalInfo)
    {
        return MockFunctionTest::aclStubInstance().GetPlatformInfos(SoCVersion, platformInfo, optionalInfo);
    }

    bool PlatFormInfos::GetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res) {
        return MockFunctionTest::aclStubInstance().GetPlatformResWithLock(label, res);
    }

    bool PlatFormInfos::GetPlatformResWithLock(const string &label, const string &key, string &val)
    {
        return MockFunctionTest::aclStubInstance().GetPlatformResWithLock(label, key, val);
    }

    void PlatFormInfos::SetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res) {
        return;
    }
}

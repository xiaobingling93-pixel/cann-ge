/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/executor/ge_executor.h"
#include "framework/generator/ge_generator.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/mem_allocator.h"
#include "framework/runtime/stream_executor.h"
#include "framework/runtime/gert_api.h"
#include "framework/memory/allocator_desc.h"
#include "framework/runtime/gert_api.h"
#include "framework/runtime/om2_model_executor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_tensor.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/model.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/operator.h"
#include "ge/ge_api.h"
#include "common/ge_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/opsproto_manager.h"
#include "graph/operator_factory.h"
#include "graph/ge_local_context.h"
#include "graph/tensor.h"
#include "common/helper/om_file_helper.h"
#include "platform/platform_info.h"

#include "runtime/kernel.h"

#include "adx_datadump_server.h"
#include "adump_api.h"
#include "mmpa/mmpa_api.h"

#include "acl/acl_rt.h"
#include "acl/acl_op.h"
#include "acl/acl_rt_allocator.h"

#include <gmock/gmock.h>

// using namespace tdt;
using namespace ge;

typedef aclError (*aclDumpSetCallbackFunc)(const char *configStr);
typedef aclError (*aclDumpUnsetCallbackFunc)();

class aclStub
{
public:
    // error manager
    virtual std::unique_ptr<const char_t[]> GetErrMgrErrorMessage();

    // ge function
    virtual ge::Status SetDump(const ge::DumpConfig &dumpConfig);
    virtual ge::Status GEInitialize(const std::map<AscendString, AscendString> &options);
    virtual ge::Status Finalize();
    virtual ge::Status Ge_Generator_Finalize();
    virtual ge::Status GEFinalize();
    virtual ge::Status BuildSingleOpModel(ge::OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                          const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                          int32_t compile_flag, ModelBufferData &model_buff);
    virtual ge::Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                          const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                          int32_t compile_flag, ModelBufferData &model_buff,
                                          GraphStage graph_stage, ComputeGraphPtr &compute_graph);
    virtual graphStatus SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range);
    virtual bool ReadBytesFromBinaryFile(char const *file_name, char **buffer, int &length);
    virtual ge::Status Initialize(const std::map<std::string, std::string> &options);
    virtual ge::Status Initialize(const std::map<std::string, std::string> &options, OmgContext &omgContext);
    virtual ge::Status LoadSingleOpV2(const std::string &modelName,
                                      const ModelData &modelData,
                                      void *stream,
                                      SingleOp **single_op,
                                      const uint64_t model_id);
    virtual ge::Status LoadDynamicSingleOpV2(const std::string &model_name,
                                             const ge::ModelData &modelData,
                                             void *stream,
                                             DynamicSingleOp **single_op,
                                             const uint64_t model_id);
    virtual ge::Status ExecuteAsync(DynamicSingleOp *executor,
                                    const std::vector<GeTensorDesc> &input_desc,
                                    const std::vector<DataBuffer> &inputs,
                                    std::vector<GeTensorDesc> &output_desc,
                                    std::vector<DataBuffer> &outputs);
    virtual ge::Status ExecuteAsync(SingleOp *executor,
                                    const std::vector<DataBuffer> &inputs,
                                    std::vector<DataBuffer> &outputs);
    virtual graphStatus GetName(AscendString &name);
    virtual bool GetBool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, bool &value);
    virtual bool GetInt(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &name, int32_t &value);
    virtual bool GetListNamedAttrs(AttrUtils::ConstAttrHolderAdapter &&obj, std::string const &name, vector<GeAttrValue::NAMED_ATTRS> &value);
    virtual std::map<string, AnyValue> GetAllAttrs();
    virtual std::string RealPath(const char *path);
    virtual graphStatus GetOpsTypeList(std::vector<ge::AscendString> &all_ops);
    virtual ge::Status GetModelDescInfo(uint32_t modelId, std::vector<TensorDesc> &inputDesc,
                                        std::vector<TensorDesc> &outputDesc, bool new_model_desc);

    virtual ge::Status GetModelDescInfoFromMem(const ModelData &model_data, ModelInOutInfo &info);
    virtual graphStatus GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range);
    virtual Format GetFormat();
    virtual ge::Status GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                           int32_t &dynamic_type);
    virtual ge::Status LoadModelFromData(uint32_t &model_id, const ModelData &modelData,
                                         void *dev_ptr, size_t memsize, void *weight_ptr, size_t weightsize);
    virtual ge::Status LoadModelFromDataWithArgs(uint32_t &model_id, const ModelData &model_data, const ModelLoadArg &load_arg);
    virtual ge::graphStatus LoadDataFromFile(std::string const &path, ModelData &modelData);
    virtual ge::Status LoadModelWithQ(uint32_t &model_id, const ge::ModelData &ge_model_data,
                                      const std::vector<uint32_t> &input_queue_ids, const std::vector<uint32_t> &output_queue_ids);
    virtual ge::Status LoadModelWithQ(uint32_t &model_id, const ge::ModelData &ge_model_data,
                                      const ge::ModelQueueArg &queue_arg);
    virtual ge::Status UnloadModel(uint32_t modelId);
    virtual ge::Status GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size);
    virtual ge::Status GetMemAndWeightSize(const void *model_data, size_t model_size, size_t &mem_size, size_t &weight_size);
    virtual ge::Status ExecModel(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                                 const std::vector<ge::GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                                 std::vector<ge::GeTensorDesc> &output_desc, bool async_mode);
    virtual ge::Status SetDynamicBatchSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t batch_size);
    virtual ge::Status SetDynamicImageSize(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t image_height, uint64_t image_width);
    virtual ge::Status SetDynamicDims(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                      const vector<uint64_t> &dynamic_dims);
    virtual ge::Status GetCurDynamicDims(uint32_t model_id, const vector<uint64_t> &dynamic_dims,
                                         vector<uint64_t> &cur_dynamic_dims);
    virtual ge::Status GetAippType(uint32_t model_id, uint32_t index, ge::InputAippType &type, size_t &aippindex);
    virtual ge::Status GetAippType( uint32_t index, ge::InputAippType &type, size_t &aippindex);
    virtual ge::Status GetUserDesignateShapeOrder(uint32_t model_id, vector<string> &user_designate_shape_order);
    virtual ge::Status GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type);
    virtual ge::Status GetModelAttr(uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info);
    virtual ge::Status GetOpAttr(uint32_t model_id, const std::string &op_name, const std::string &attr_name, std::string &attr_value);
    virtual ge::Status GetAIPPInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_params);
    virtual ge::Status GetAippInfo(const uint32_t index, ge::AippConfigInfo &aipp_info);
    virtual ge::Status GetBatchInfoSize(uint32_t model_id, size_t &shape_count);
    virtual ge::Status GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &origOutputInfo);
    virtual ge::Status GetOriginAippInputInfo(uint32_t index, OriginInputInfo &origOutputInfo);
    virtual ge::Status GetAllAippInputOutputDims(uint32_t model_id, uint32_t index,
                                                 std::vector<InputOutputDims> &input_dims,
                                                 std::vector<InputOutputDims> &output_dims);
    virtual ge::Status GetAllAippInputOutputDims(uint32_t index,
                                                 std::vector<InputOutputDims> &input_dims,
                                                 std::vector<InputOutputDims> &output_dims);
    virtual ge::Status SetDynamicAippData(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                          const std::vector<kAippDynamicBatchPara> &aippBatchPara,
                                          const kAippDynamicPara &aippParms);
    virtual int Init();
    virtual bool OpsProtoManager_Initialize(const std::map<std::string, std::string> &options);
    virtual ge::Status TransShape(const TensorDesc &src_desc,
                                  Format dst_format,
                                  std::vector<int64_t> &dst_shape);
    virtual ge::Status Init(uint8_t *model_data, const uint32_t model_data_size);
    virtual ge::Status GetModelPartition(ModelPartitionType type, ModelPartition &partition);
    virtual graphStatus Load(const uint8_t *data, size_t len, Model &model);
    virtual bool HasAttr(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name);
    virtual bool GetListTensor(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, vector<ConstGeTensorPtr> &value);
    virtual bool IsOriginShapeInRange(const gert::Shape &shape);
    virtual ge::Status SetAllocator(void *const stream, ge::Allocator *const external_allocator);

    // RT2.0
    virtual gert::ModelV2Executor *GetOrCreateLoaded(rtStream_t stream, const gert::ModelExecuteArg &arg);
    virtual gert::ModelV2Executor *CreateAndLoad(rtStream_t stream, const gert::ModelExecuteArg &arg);
    virtual ge::graphStatus Erase(rtStream_t stream);
    virtual std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromFile(const char *file_path, ge::graphStatus &error_code);
    virtual std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data, ge::graphStatus &error_code);
    virtual std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromModelDataWithRtSession(const ge::ModelData &model_data,
                                                                                          gert::RtSession *const rt_session,
                                                                                          ge::graphStatus &error_code);
    virtual ge::graphStatus LoadDataFromFileV2(const char *path, ge::ModelData &model_data);
    virtual std::unique_ptr<gert::ModelV2Executor>
    LoadExecutorFromModelDataWithMem(const ge::ModelData &model_data, ge::graphStatus &error_code,
                                     const void *weight_ptr, const size_t weight_size);
    virtual std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data, const void *weight_ptr,
                                                                                  const size_t weight_size, ge::graphStatus &error_code);
    virtual std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                                  const gert::LoweringOption &optimize_option,
                                                                                  ge::graphStatus &error_code);
    virtual ge::graphStatus IsDynamicModel(const char *file_path, bool &is_dynamic_model);
    virtual ge::graphStatus Load();
    virtual ge::graphStatus Load(const gert::ModelExecuteArg &arg);
    virtual ge::graphStatus Load(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg);
    virtual std::unique_ptr<ge::Allocator> Create(const gert::TensorPlacement &placement);
    virtual ge::graphStatus Execute(const gert::ModelExecuteArg &arg,
                                    gert::Tensor **inputs, size_t input_num,
                                    gert::Tensor **outputs, size_t output_num);
    virtual ge::graphStatus ExecuteSync(gert::Tensor **inputs, size_t input_num,
                                        gert::Tensor **outputs, size_t output_num);
    virtual ge::graphStatus UnLoad();

    // OM2
    virtual ge::Status LoadOm2DataFromFile(const std::string &model_path, ge::ModelData &model_data);
    virtual std::unique_ptr<gert::Om2ModelExecutor> LoadOm2ExecutorFromData(ge::ModelData &model_data,
                                                                            ge::Status &error_code);
    virtual ge::Status IsOm2Model(const void *data, size_t size, bool &is_support);
    virtual ge::Status IsOm2Model(const char *file_path, bool &is_support);
    virtual ge::Status GetModelDescInfo(std::vector<ge::TensorDesc> &input_desc,
                                        std::vector<ge::TensorDesc> &output_desc, bool new_model_desc);

        // fe function
        virtual uint32_t InitializePlatformInfo();
    virtual uint32_t GetPlatformInfos(
        const std::string SoCVersion, fe::PlatFormInfos &platformInfo, fe::OptionalInfos &optionalInfo);
    virtual uint32_t InitRuntimePlatformInfos(const std::string &SoCVersion);
    virtual uint32_t GetRuntimePlatformInfosByDevice(const uint32_t &device_id, fe::PlatFormInfos &platform_infos);
    virtual uint32_t UpdateRuntimePlatformInfosByDevice(const uint32_t &device_id,
                                                        fe::PlatFormInfos &platform_infos);
    virtual bool GetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res);
    virtual bool GetPlatformResWithLock(const string &label, const string &key, string &val);

    // runtime
    virtual rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                     rtSmDesc_t *smDesc, rtStream_t stream);
    virtual rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName,
                                         const void *devFunc, uint32_t funcMode);
    virtual rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle);
    virtual rtError_t rtDevBinaryUnRegister(void *handle);
    virtual rtError_t rtGetSocSpec(const char *label, const char *key, char *value, const uint32_t maxLen);

    // prof function
    virtual int32_t MsprofFinalize();
    virtual int32_t MsprofInit(uint32_t aclDataType, void *data, uint32_t dataLen);
    virtual int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName);
    // adx function
    virtual int AdxDataDumpServerInit();
    virtual int AdxDataDumpServerUnInit();
    virtual int32_t AdumpSetDumpConfig(Adx::DumpType dumpType, const Adx::DumpConfig &dumpConfig);
    virtual bool AdumpIsDumpEnable(Adx::DumpType dumpType);

    // slog function
    virtual int dlog_getlevel(int module_id, int *enable_event);

    // mmpa function
    virtual void *mmAlignMalloc(mmSize mallocSize, mmSize alignSize);
    virtual INT32 mmAccess2(const CHAR *pathName, INT32 mode);
    virtual INT32 mmDladdr(VOID *addr, mmDlInfo *info);

    // acl_rt
    virtual aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);
    virtual aclError aclrtFree(void *devPtr);
    virtual aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
    virtual aclError aclrtGetEventId(aclrtEvent event, uint32_t *eventId);
    virtual aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream);
    virtual aclError aclrtDestroyEvent(aclrtEvent event);
    virtual aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);
    virtual aclError aclrtGetRunMode(aclrtRunMode *runMode);
    virtual aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
    virtual aclError aclrtCreateStream(aclrtStream *stream);
    virtual aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
        aclrtMemcpyKind kind, aclrtStream stream);
    virtual aclError aclrtDestroyStream(aclrtStream stream);
    virtual aclError aclrtSynchronizeStream(aclrtStream stream);
    virtual aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t *notifyId);
    virtual aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream);
    virtual aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream);
    virtual aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);
    virtual aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority);
    virtual aclError aclrtCtxGetCurrentDefaultStream(aclrtStream *stream);
    virtual aclError aclrtRegStreamStateCallback(const char *regName, aclrtStreamStateCallback callback, void *args);
    virtual aclError aclrtRegDeviceStateCallback(const char *regName, aclrtDeviceStateCallback callback, void *args);
    virtual aclError aclrtLaunchCallback(aclrtCallback fn, void *userData,
        aclrtCallbackBlockType blockType, aclrtStream stream);
    virtual aclError aclrtGetDevice(int32_t *deviceId);
    virtual aclDataBuffer *aclCreateDataBuffer(void *data, size_t size);
    virtual void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer);
    virtual aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer);
    // virtual hi_s32 hi_mpi_vpc_equalize_hist_for_acl(hi_s32 chn, const hi_vpc_pic_info* source_pic,
    //     hi_vpc_pic_info* dest_pic, const hi_vpc_lut_remap* hist_config_ptr, DvppStream stream);
    virtual size_t aclDataTypeSize(aclDataType dataType);
    virtual aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout);
    virtual aclError aclrtAllocatorGetByStream(aclrtStream stream,
                                    aclrtAllocatorDesc *allocatorDesc,
                                    aclrtAllocator *allocator,
                                    aclrtAllocatorAllocFunc *allocFunc,
                                    aclrtAllocatorFreeFunc *freeFunc,
                                    aclrtAllocatorAllocAdviseFunc *allocAdviseFunc,
                                    aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc);
    virtual aclError aclInitCallbackRegister(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc,
                                                            void *userData);
    virtual aclError aclInitCallbackUnRegister(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc);
    virtual aclError aclFinalizeCallbackRegister(aclRegisterCallbackType type,
                                                                aclFinalizeCallbackFunc cbFunc, void *userData);
    virtual aclError aclFinalizeCallbackUnRegister(aclRegisterCallbackType type,
                                                                aclFinalizeCallbackFunc cbFunc);
    virtual size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer);
    virtual uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer);
    virtual const char *aclrtGetSocName();
    virtual aclError aclDumpSetCallbackRegister(aclDumpSetCallbackFunc cbFunc);
    virtual aclError aclDumpSetCallbackUnRegister();
    virtual aclError aclDumpUnsetCallbackRegister(aclDumpUnsetCallbackFunc cbFunc);
    virtual aclError aclDumpUnsetCallbackUnRegister();
    virtual aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue);
    virtual aclError aclrtGetCurrentContext(aclrtContext *context);
    virtual aclError aclrtSetCurrentContext(aclrtContext context);

    // mmpa
    virtual INT32 mmGetTid();
};

class MockFunctionTest : public aclStub
{
public:
    MockFunctionTest();
    static MockFunctionTest &aclStubInstance();
    void ResetToDefaultMock();

    // error manager
    MOCK_METHOD0(GetErrMgrErrorMessage, std::unique_ptr<const char_t[]>());

    // ge function stub
    MOCK_METHOD1(SetDump, ge::Status(const ge::DumpConfig &dump_config));
    MOCK_METHOD1(GEInitialize, ge::Status(const std::map<AscendString, AscendString> &options));
    MOCK_METHOD0(Finalize, ge::Status());
    MOCK_METHOD0(GEFinalize, ge::Status());
    MOCK_METHOD6(BuildSingleOpModel, ge::Status(ge::OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                                const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                                int32_t compile_flag, ModelBufferData &model_buff));
    MOCK_METHOD8(BuildSingleOpModel, ge::Status(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                                const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                                                int32_t compile_flag, ModelBufferData &model_buff,
                                                GraphStage graph_stage, ComputeGraphPtr &compute_graph));
    MOCK_METHOD1(SetShapeRange, graphStatus(const std::vector<std::pair<int64_t, int64_t>> &range));
    MOCK_METHOD3(ReadBytesFromBinaryFile, bool(char const *file_name, char **buffer, int &length));
    MOCK_METHOD1(Initialize, ge::Status(const std::map<std::string, std::string> &options));
    MOCK_METHOD1(GetName, graphStatus(AscendString &name));
    MOCK_METHOD2(Initialize, ge::Status(const std::map<std::string, std::string> &options, OmgContext &omgContext));
    MOCK_METHOD0(Ge_Generator_Finalize, ge::Status());
    MOCK_METHOD5(LoadSingleOpV2, ge::Status(const std::string &modelName, const ModelData &modelData, void *stream,
                                            SingleOp **single_op, const uint64_t model_id));
    MOCK_METHOD2(SetAllocator, ge::Status(void *const stream, ge::Allocator *const external_allocator));
    MOCK_METHOD5(LoadDynamicSingleOpV2, ge::Status(const std::string &model_name, const ge::ModelData &modelData, void *stream,
                                                   DynamicSingleOp **single_op, const uint64_t model_id));
    MOCK_METHOD5(ExecuteAsync, ge::Status(DynamicSingleOp *executor, const std::vector<GeTensorDesc> &input_desc,
                                          const std::vector<DataBuffer> &inputs, std::vector<GeTensorDesc> &output_desc, std::vector<DataBuffer> &outputs));
    MOCK_METHOD3(ExecuteAsync, ge::Status(SingleOp *executor, const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs));
    MOCK_METHOD3(GetBool, bool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, bool &value));
    MOCK_METHOD3(GetInt, bool(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &name, int32_t &value));
    MOCK_METHOD3(GetListNamedAttrs, bool(ge::AttrUtils::ConstAttrHolderAdapter &&obj, std::string const &name, vector<GeAttrValue::NAMED_ATTRS> &value));
    MOCK_METHOD0(GetAllAttrs, std::map<string, AnyValue>());
    MOCK_METHOD1(RealPath, std::string(const char *path));
    MOCK_METHOD1(GetOpsTypeList, graphStatus(std::vector<ge::AscendString> &all_ops));
    MOCK_METHOD4(GetModelDescInfo, ge::Status(uint32_t modelId, std::vector<TensorDesc> &inputDesc,
                                              std::vector<TensorDesc> &outputDesc, bool new_model_desc));

    MOCK_METHOD2(GetModelDescInfoFromMem, ge::Status(const ModelData &model_data, ModelInOutInfo &info));
    MOCK_METHOD1(GetShapeRange, graphStatus(std::vector<std::pair<int64_t, int64_t>> &range));
    MOCK_METHOD0(GetFormat, Format());
    MOCK_METHOD3(GetDynamicBatchInfo, ge::Status(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type));
    MOCK_METHOD6(LoadModelFromData, ge::Status(uint32_t &model_id, const ModelData &modelData,
                                               void *dev_ptr, size_t memsize, void *weight_ptr, size_t weightsize));
    MOCK_METHOD3(LoadModelFromDataWithArgs, ge::Status(uint32_t &model_id, const ModelData &model_data, const ModelLoadArg &load_arg));
    MOCK_METHOD2(LoadDataFromFile, ge::graphStatus(std::string const &path, ModelData &modelData));
    MOCK_METHOD4(LoadModelWithQ, ge::Status(uint32_t &model_id, const ge::ModelData &ge_model_data,
                                            const std::vector<uint32_t> &input_queue_ids,
                                            const std::vector<uint32_t> &output_queue_ids));
    MOCK_METHOD3(LoadModelWithQ, ge::Status(uint32_t &model_id, const ge::ModelData &ge_model_data,
                                            const ge::ModelQueueArg &queue_arg));
    MOCK_METHOD1(UnloadModel, ge::Status(uint32_t modelId));
    MOCK_METHOD3(GetMemAndWeightSize, ge::Status(const std::string &path, size_t &mem_size, size_t &weight_size));
    MOCK_METHOD4(GetMemAndWeightSize, ge::Status(const void *model_data, size_t model_size, size_t &mem_size, size_t &weight_size));
    MOCK_METHOD7(ExecModel, ge::Status(uint32_t model_id, void *stream, const ge::RunModelData &run_input_data,
                                       const std::vector<ge::GeTensorDesc> &input_desc, ge::RunModelData &run_output_data,
                                       std::vector<ge::GeTensorDesc> &output_desc, bool async_mode));
    MOCK_METHOD4(SetDynamicBatchSize, ge::Status(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t batch_size));
    MOCK_METHOD5(SetDynamicImageSize, ge::Status(uint32_t model_id, void *dynamic_input_addr, uint64_t length, uint64_t image_height, uint64_t image_width));
    MOCK_METHOD4(SetDynamicDims, ge::Status(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                            const vector<uint64_t> &dynamic_dims));
    MOCK_METHOD3(GetCurDynamicDims, ge::Status(uint32_t model_id, const vector<uint64_t> &dynamic_dims,
                                               vector<uint64_t> &cur_dynamic_dims));
    MOCK_METHOD4(GetAippType, ge::Status(uint32_t model_id, uint32_t index, ge::InputAippType &type, size_t &aippindex));
    MOCK_METHOD3(GetAippType, ge::Status(uint32_t index, ge::InputAippType &type, size_t &aippindex));
    MOCK_METHOD2(GetUserDesignateShapeOrder, ge::Status(uint32_t model_id, vector<string> &user_designate_shape_order));
    MOCK_METHOD3(GetCurShape, ge::Status(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type));
    MOCK_METHOD2(GetModelAttr, ge::Status(uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info));
    MOCK_METHOD4(GetOpAttr, ge::Status(uint32_t model_id, const std::string &op_name, const std::string &attr_name, std::string &attr_value));
    MOCK_METHOD3(GetAIPPInfo, ge::Status(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_params));
    MOCK_METHOD2(GetAippInfo, ge::Status(const uint32_t index, ge::AippConfigInfo &aipp_info));
    MOCK_METHOD2(GetBatchInfoSize, ge::Status(uint32_t model_id, size_t &shape_count));
    MOCK_METHOD3(GetOrigInputInfo, ge::Status(uint32_t model_id, uint32_t index, OriginInputInfo &origOutputInfo));
    MOCK_METHOD2(GetOriginAippInputInfo, ge::Status(uint32_t index, OriginInputInfo &origOutputInfo));
    MOCK_METHOD4(GetAllAippInputOutputDims, ge::Status(uint32_t model_id, uint32_t index, std::vector<InputOutputDims> &input_dims, std::vector<InputOutputDims> &output_dims));
    MOCK_METHOD3(GetAllAippInputOutputDims, ge::Status(uint32_t index, std::vector<InputOutputDims> &input_dims, std::vector<InputOutputDims> &output_dims));
    MOCK_METHOD5(SetDynamicAippData, ge::Status(uint32_t model_id, void *dynamic_input_addr, uint64_t length,
                                                const std::vector<kAippDynamicBatchPara> &aippBatchPara,
                                                const kAippDynamicPara &aippParms));
    MOCK_METHOD0(Init, int());
    MOCK_METHOD1(OpsProtoManager_Initialize, bool(const std::map<std::string, std::string> &options));
    MOCK_METHOD3(TransShape, ge::Status(const TensorDesc &src_desc, Format dst_format,
                                        std::vector<int64_t> &dst_shape));
    MOCK_METHOD3(Load, graphStatus(const uint8_t *data, size_t len, Model &model));
    MOCK_METHOD2(Init, ge::Status(uint8_t *model_data, const uint32_t model_data_size));
    MOCK_METHOD2(GetModelPartition, ge::Status(ModelPartitionType type, ModelPartition &partition));
    MOCK_METHOD2(HasAttr, bool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name));
    MOCK_METHOD3(GetListTensor,
        bool(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, vector<ConstGeTensorPtr> &value));
    MOCK_METHOD1(IsOriginShapeInRange, bool(const gert::Shape &shape));

    // RT2.0 function stub
    MOCK_METHOD2(GetOrCreateLoaded, gert::ModelV2Executor *(rtStream_t stream, const gert::ModelExecuteArg &arg));
    MOCK_METHOD2(CreateAndLoad, gert::ModelV2Executor *(rtStream_t stream, const gert::ModelExecuteArg &arg));
    MOCK_METHOD1(Erase, ge::graphStatus(rtStream_t stream));
    MOCK_METHOD2(LoadExecutorFromFile, std::unique_ptr<gert::ModelV2Executor>(const char *file_path, ge::graphStatus &error_code));
    MOCK_METHOD2(LoadExecutorFromModelData, std::unique_ptr<gert::ModelV2Executor>(const ge::ModelData &model_data,
                                                                                   ge::graphStatus &error_code));
    MOCK_METHOD3(LoadExecutorFromModelData, std::unique_ptr<gert::ModelV2Executor>(const ge::ModelData &model_data,
                                                                                   const gert::LoadExecutorArgs &args,
                                                                                   ge::graphStatus &error_code));
    MOCK_METHOD3(LoadExecutorFromModelDataWithRtSession, std::unique_ptr<gert::ModelV2Executor> (const ge::ModelData &model_data,
                                                                                  gert::RtSession *const rt_session,
                                                                                  ge::graphStatus &error_code));
    MOCK_METHOD2(LoadDataFromFileV2, ge::graphStatus(const char *path, ge::ModelData &modelData));
    MOCK_METHOD4(LoadExecutorFromModelDataWithMem, std::unique_ptr<gert::ModelV2Executor>(
                                                       const ge::ModelData &model_data, ge::graphStatus &error_code, const void *weight_ptr,
                                                       const size_t weight_size));
    MOCK_METHOD4(LoadStreamExecutorFromModelData, std::unique_ptr<gert::StreamExecutor>(
                                                      const ge::ModelData &model_data, const void *weight_ptr, const size_t weight_size, ge::graphStatus &error_code));
    MOCK_METHOD3(LoadStreamExecutorFromModelData, std::unique_ptr<gert::StreamExecutor>(
                                                      const ge::ModelData &model_data, const gert::LoweringOption &optimize_option, ge::graphStatus &error_code));
    MOCK_METHOD2(IsDynamicModel, ge::graphStatus(const char *file_path, bool &is_dynamic_model));
    MOCK_METHOD0(Load, ge::graphStatus());
    MOCK_METHOD1(Load, ge::graphStatus(const gert::ModelExecuteArg &arg));
    MOCK_METHOD2(Load, ge::graphStatus(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg));
    MOCK_METHOD1(Create, std::unique_ptr<ge::Allocator>(const gert::TensorPlacement &placement));
    MOCK_METHOD5(Execute, ge::graphStatus(const gert::ModelExecuteArg &arg,
                                          gert::Tensor **inputs, size_t input_num,
                                          gert::Tensor **outputs, size_t output_num));
    MOCK_METHOD4(ExecuteSync, ge::graphStatus(gert::Tensor **inputs, size_t input_num,
                                              gert::Tensor **outputs, size_t output_num));

    MOCK_METHOD0(UnLoad, ge::graphStatus());

    // OM2
    MOCK_METHOD2(LoadOm2DataFromFile, ge::Status(const std::string &model_path, ge::ModelData &model_data));
    MOCK_METHOD2(LoadOm2ExecutorFromData,
                 std::unique_ptr<gert::Om2ModelExecutor>(ge::ModelData &model_data, ge::Status &error_code));
    MOCK_METHOD3(IsOm2Model, ge::Status(const void *data, size_t size, bool &is_support));
    MOCK_METHOD2(IsOm2Model, ge::Status(const char *file_path, bool &is_support));
    MOCK_METHOD3(GetModelDescInfo, ge::Status(std::vector<ge::TensorDesc> &input_desc,
                                              std::vector<ge::TensorDesc> &output_desc, bool new_model_desc));

    // fe function
    MOCK_METHOD0(InitializePlatformInfo, uint32_t());
    MOCK_METHOD3(GetPlatformInfos,
        uint32_t(const std::string SoCVersion, fe::PlatFormInfos &platformInfo, fe::OptionalInfos &optionalInfo));
    MOCK_METHOD1(InitRuntimePlatformInfos, uint32_t(const std::string &SoCVersion));
    MOCK_METHOD2(GetRuntimePlatformInfosByDevice, uint32_t(const uint32_t &device_id, fe::PlatFormInfos &platform_infos));
    MOCK_METHOD2(GetPlatformResWithLock, bool(const std::string &label, std::map<std::string, std::string> &res));
    MOCK_METHOD3(GetPlatformResWithLock, bool(const string &label, const string &key, string &val));
    MOCK_METHOD2(UpdateRuntimePlatformInfosByDevice, uint32_t(const uint32_t &device_id, fe::PlatFormInfos &platform_infos));

    // runtime
    MOCK_METHOD6(rtKernelLaunch, rtError_t(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                           rtSmDesc_t *smDesc, rtStream_t stream));
    MOCK_METHOD5(rtFunctionRegister, rtError_t(void *binHandle, const void *stubFunc, const char *stubName,
                                               const void *devFunc, uint32_t funcMode));
    MOCK_METHOD1(rtDevBinaryUnRegister, rtError_t(void *handle));
    MOCK_METHOD2(rtDevBinaryRegister, rtError_t(const rtDevBinary_t *bin, void **handle));
    MOCK_METHOD4(rtGetSocSpec, rtError_t(const char *label, const char *key, char *value, const uint32_t maxLen));

    // prof function stub
    MOCK_METHOD0(MsprofFinalize, int32_t());
    MOCK_METHOD3(MsprofInit, int32_t(uint32_t aclDataType, void *data, uint32_t dataLen));
    MOCK_METHOD3(MsprofRegTypeInfo, int32_t(uint16_t level, uint32_t typeId, const char *typeName));

    // adx function stub
    MOCK_METHOD0(AdxDataDumpServerInit, int());
    MOCK_METHOD0(AdxDataDumpServerUnInit, int());
    MOCK_METHOD2(AdumpSetDumpConfig, int(Adx::DumpType dumpType, const Adx::DumpConfig &dumpConfig));
    MOCK_METHOD1(AdumpIsDumpEnable, bool(Adx::DumpType dumpType));

    // slog function stub
    MOCK_METHOD2(dlog_getlevel, int(int module_id, int *enable_event));

    // mmpa function stub
    MOCK_METHOD2(mmAlignMalloc, void *(mmSize mallocSize, mmSize alignSize));
    MOCK_METHOD2(mmAccess2, INT32(const CHAR *pathName, INT32 mode));
    MOCK_METHOD2(mmDladdr, INT32(VOID *addr, mmDlInfo *info));

    // acl_rt
    MOCK_METHOD2(aclrtCreateEventWithFlag, aclError(aclrtEvent *event, uint32_t flag));
    MOCK_METHOD1(aclrtFree, aclError(void *devPtr));
    MOCK_METHOD3(aclrtMalloc, aclError(void **devPtr, size_t size, aclrtMemMallocPolicy policy));
    MOCK_METHOD2(aclrtGetEventId, aclError(aclrtEvent event, uint32_t *eventId));
    MOCK_METHOD2(aclrtResetEvent, aclError(aclrtEvent event, aclrtStream stream));
    MOCK_METHOD1(aclrtDestroyEvent, aclError(aclrtEvent event));
    MOCK_METHOD2(aclrtStreamWaitEvent, aclError(aclrtStream stream, aclrtEvent event));
    MOCK_METHOD1(aclrtGetRunMode, aclError(aclrtRunMode *runMode));
    MOCK_METHOD5(aclrtMemcpy, aclError(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind));
    MOCK_METHOD1(aclrtCreateStream, aclError(aclrtStream *stream));
    MOCK_METHOD6(aclrtMemcpyAsync, aclError(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream));
    MOCK_METHOD1(aclrtDestroyStream, aclError(aclrtStream stream));
    MOCK_METHOD1(aclrtSynchronizeStream, aclError(aclrtStream stream));
    MOCK_METHOD2(aclrtGetNotifyId, aclError(aclrtNotify notify, uint32_t *notifyId));
    MOCK_METHOD2(aclrtUnSubscribeReport, aclError(uint64_t threadId, aclrtStream stream));
    MOCK_METHOD2(aclrtSubscribeReport, aclError(uint64_t threadId, aclrtStream stream));
    MOCK_METHOD4(aclrtMemset, aclError(void *devPtr, size_t maxCount, int32_t value, size_t count));
    MOCK_METHOD2(aclrtDeviceGetStreamPriorityRange, aclError(int32_t *leastPriority, int32_t *greatestPriority));
    MOCK_METHOD1(aclrtCtxGetCurrentDefaultStream, aclError(aclrtStream *stream));
    MOCK_METHOD3(aclrtRegStreamStateCallback, aclError(const char *regName, aclrtStreamStateCallback callback, void *args));
    MOCK_METHOD3(aclrtRegDeviceStateCallback, aclError(const char *regName, aclrtDeviceStateCallback callback, void *args));
    MOCK_METHOD4(aclrtLaunchCallback, aclError(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, aclrtStream stream));
    MOCK_METHOD1(aclrtGetDevice, aclError(int32_t *deviceId));
    MOCK_METHOD2(aclCreateDataBuffer, aclDataBuffer *(void *data, size_t size));
    MOCK_METHOD1(aclGetDataBufferAddr, void *(const aclDataBuffer *dataBuffer));
    MOCK_METHOD1(aclDestroyDataBuffer, aclError(const aclDataBuffer *dataBuffer));
    MOCK_METHOD1(aclDataTypeSize, size_t(aclDataType dataType));
    MOCK_METHOD2(aclrtSynchronizeStreamWithTimeout, aclError(aclrtStream stream, int32_t timeout));
    MOCK_METHOD7(aclrtAllocatorGetByStream, aclError(aclrtStream stream,
                                    aclrtAllocatorDesc *allocatorDesc,
                                    aclrtAllocator *allocator,
                                    aclrtAllocatorAllocFunc *allocFunc,
                                    aclrtAllocatorFreeFunc *freeFunc,
                                    aclrtAllocatorAllocAdviseFunc *allocAdviseFunc,
                                    aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc));
    MOCK_METHOD3(aclInitCallbackRegister, aclError(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc,
                                                            void *userData));
    MOCK_METHOD2(aclInitCallbackUnRegister, aclError(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc));
    MOCK_METHOD3(aclFinalizeCallbackRegister, aclError(aclRegisterCallbackType type,
                                                                aclFinalizeCallbackFunc cbFunc, void *userData));
    MOCK_METHOD2(aclFinalizeCallbackUnRegister, aclError(aclRegisterCallbackType type,
                                                                aclFinalizeCallbackFunc cbFunc));
    MOCK_METHOD1(aclGetDataBufferSizeV2, size_t(const aclDataBuffer *dataBuffer));
    MOCK_METHOD1(aclGetDataBufferSize, uint32_t(const aclDataBuffer *dataBuffer));
    MOCK_METHOD0(aclrtGetSocName, const char *());
    MOCK_METHOD1(aclDumpSetCallbackRegister, aclError(aclDumpSetCallbackFunc cbFunc));
    MOCK_METHOD0(aclDumpSetCallbackUnRegister, aclError());
    MOCK_METHOD1(aclDumpUnsetCallbackRegister, aclError(aclDumpUnsetCallbackFunc cbFunc));
    MOCK_METHOD0(aclDumpUnsetCallbackUnRegister, aclError());
    MOCK_METHOD3(aclopSetAttrBool, aclError(aclopAttr *attr, const char *attrName, uint8_t attrValue));
    MOCK_METHOD1(aclrtGetCurrentContext, aclError(aclrtContext *context));
    MOCK_METHOD1(aclrtSetCurrentContext, aclError(aclrtContext context));

    // mmpa
    MOCK_METHOD0(mmGetTid, INT32());
};

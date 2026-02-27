/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_executor.h"
#include <numeric>
#include "securec.h"
#include "common/common_define.h"
#include "common/inner_error_codes.h"
#include "common/udf_log.h"
#include "common/util.h"
#include "common/scope_guard.h"
#include "config/global_config.h"
#include "flow_func/flow_func_timer.h"
#include "flow_func/logger/flow_func_logger_manager.h"
#include "flow_func/flow_func_manager.h"
#include "flow_model_impl.h"
#include "flow_func_drv_manager.h"
#include "memory_statistic_manager.h"

namespace FlowFunc {
namespace {
// drv max value is 10s
constexpr int32_t kAttachWaitTimeout = 10 * 1000;
constexpr int32_t kAsyncExecutorThreadNum = 1;
constexpr uint64_t kWaitEventMask =
    (1ULL << UdfEvent::kEventIdProcessorInit) |
    (1ULL << UdfEvent::kEventIdFlowFuncInit) |
    (1ULL << static_cast<uint32_t>(EVENT_QUEUE_EMPTY_TO_NOT_EMPTY)) |
    (1ULL << static_cast<uint32_t>(EVENT_QUEUE_FULL_TO_NOT_FULL)) |
    (1ULL << UdfEvent::kEventIdTimer) |
    (1ULL << UdfEvent::kEventIdFlowFuncReportStatus) |
    (1ULL << UdfEvent::kEventIdNotifyThreadExit) |
    (1ULL << UdfEvent::kEventIdFlowFuncSuspendFinished) |
    (1ULL << UdfEvent::kEventIdFlowFuncRecoverFinished) |
    (1ULL << UdfEvent::kEventIdSwitchToSoftSchedMode) |
    (1ULL << UdfEvent::kEventIdRaiseException) |
    (1ULL << UdfEvent::kEventIdSingleFlowFuncInit);

constexpr uint64_t kWorkerWaitEventMask =
    (1ULL << UdfEvent::kEventIdFlowFuncExecute) |
    (1ULL << UdfEvent::kEventIdNotifyThreadExit);

// invoke model group subscribe event
constexpr uint64_t kInvokeModelWaitEventMask =
    (1ULL << static_cast<uint32_t>(EVENT_QUEUE_EMPTY_TO_NOT_EMPTY)) |
    (1ULL << static_cast<uint32_t>(EVENT_QUEUE_FULL_TO_NOT_FULL)) |
    (1ULL << UdfEvent::kEventIdWakeUp);

// flow msg queue dequeue group subscribe event
constexpr uint64_t kFlowMsgQueueWaitEventMask =
    (1ULL << static_cast<uint32_t>(EVENT_QUEUE_EMPTY_TO_NOT_EMPTY));

const std::map<ControlMessageType, std::string> kResponseMsgMap = {
    {ControlMessageType::kInit, "Execute init"},
    {ControlMessageType::kSuspend, "Execute suspend"},
    {ControlMessageType::kRecover, "Execute recover"},
    {ControlMessageType::kException, "Execute exception message"},
    {ControlMessageType::kUnknow, "Parse control message"},
};
}

FlowFuncExecutor::FlowFuncExecutor() : udf_thread_pool_("udf_worker_"),
    event_proc_func_map_(
        {{EVENT_QUEUE_EMPTY_TO_NOT_EMPTY,                &FlowFuncExecutor::ProcessEmptyToNotEmptyEvent},
         {EVENT_QUEUE_FULL_TO_NOT_FULL,                  &FlowFuncExecutor::ProcessFullToNoFullEvent},
         {UdfEvent::kEventIdProcessorInit,             &FlowFuncExecutor::ProcessProcessorInitEvent},
         {UdfEvent::kEventIdFlowFuncInit,             &FlowFuncExecutor::ProcessFlowFuncInitEvent},
         {UdfEvent::kEventIdFlowFuncExecute,          &FlowFuncExecutor::ProcessFlowFuncExecuteEvent},
         {UdfEvent::kEventIdTimer,                      &FlowFuncExecutor::ProcessTimerEvent},
         {UdfEvent::kEventIdFlowFuncReportStatus,    &FlowFuncExecutor::ProcessReportStatusEvent},
         {UdfEvent::kEventIdNotifyThreadExit,         &FlowFuncExecutor::ProcessNotifyThreadExitEvent},
         {UdfEvent::kEventIdFlowFuncSuspendFinished, &FlowFuncExecutor::ProcessReportSuspendEvent},
         {UdfEvent::kEventIdFlowFuncRecoverFinished, &FlowFuncExecutor::ProcessReportRecoverEvent},
         {UdfEvent::kEventIdSwitchToSoftSchedMode,  &FlowFuncExecutor::ProcessSwitchSoftModeEvent},
         {UdfEvent::kEventIdRaiseException,            &FlowFuncExecutor::ProcessRaiseExceptionEvent},
         {UdfEvent::kEventIdSingleFlowFuncInit,      &FlowFuncExecutor::ProcessSingleFlowFuncInitEvent}}),
    esched_process_priority_(kUserUnsetESchedPriority),
    esched_event_priority_(kUserUnsetESchedPriority) {}

void FlowFuncExecutor::UpdatePriority(int32_t user_priority, int32_t &priority) const {
    if (user_priority != kUserUnsetESchedPriority) {
        if ((priority == kUserUnsetESchedPriority) || (user_priority < priority)) {
            priority = user_priority;
        }
    }
}

int32_t FlowFuncExecutor::CreateFlowFuncProcessor(const FlowFuncModel &model,
    const std::map<std::string, std::vector<QueueDevInfo>> &input_maps,
    const std::map<std::string, std::vector<uint32_t>> &output_index_maps,
    const std::vector<QueueDevInfo> &all_outputs_queue_infos,
    const std::shared_ptr<FlowFuncParams> &params) {
    const std::string &instance_name = model.GetModelInstanceName();
    if (input_maps.empty()) {
        UDF_LOG_ERROR("func input index size cannot be zero, instance name[%s]", instance_name.c_str());
        return FLOW_FUNC_FAILED;
    }
    for (const auto &input_map : input_maps) {
        std::vector<uint32_t> output_indexes;
        auto output_iter = output_index_maps.find(input_map.first);
        if (output_iter == output_index_maps.end()) {
            UDF_LOG_INFO("multi func has no output maps, func name=%s.", input_map.first.c_str());
            output_indexes.resize(all_outputs_queue_infos.size());
            std::iota(output_indexes.begin(), output_indexes.end(), 0);
        } else {
            output_indexes = output_iter->second;
        }
        std::shared_ptr<FlowFuncProcessor> flow_func_processor(
            new(std::nothrow) FlowFuncProcessor(params, input_map.first, input_map.second,
                all_outputs_queue_infos, output_indexes, async_executor_));
        if (flow_func_processor == nullptr) {
            UDF_LOG_ERROR("alloc FlowFuncProcessor failed, flow_func_name=%s", input_map.first.c_str());
            return FLOW_FUNC_FAILED;
        }
        flow_func_processor->SetProcessorIdx(static_cast<uint32_t>(func_processors_.size()));
        const auto &stream_input_func_names = params->GetStreamInputFuncNames();
        bool is_stream_input = (stream_input_func_names.count(input_map.first) > 0);
        for (const auto &input_queue_info : input_map.second) {
            if (is_stream_input) {
                (void)flow_msg_queues_.insert(input_queue_info.queue_id);
            }
            dev_input_queue_map_[input_queue_info.device_id].emplace(input_queue_info.queue_id, input_queue_info.is_proxy_queue);
            queue_dev_set_.emplace(input_queue_info.device_id);
            if (input_queue_info.is_proxy_queue) {
                with_proxy_queue_ = true;
                continue;
            }
            if (is_stream_input) {
                // for stream input, main thread no need proc E2NE event
                continue;
            }
            std::map<uint32_t, size_t>::const_iterator context_iter = input_to_flow_func_processor_idx_.find(
                input_queue_info.queue_id);
            if (context_iter != input_to_flow_func_processor_idx_.cend()) {
                UDF_LOG_ERROR("flow_func_name[%s]'s input_queue_id=%u is used by flow func processor %zu",
                    input_map.first.c_str(),
                    input_queue_info.queue_id,
                    context_iter->second);
                return FLOW_FUNC_FAILED;
            }
            input_to_flow_func_processor_idx_[input_queue_info.queue_id] = func_processors_.size();
        }
        std::vector<QueueDevInfo> outputs_queue_infos;
        outputs_queue_infos.reserve(output_indexes.size());
        for (const auto output_index : output_indexes) {
            const auto &output_queue_info = all_outputs_queue_infos[output_index];
            outputs_queue_infos.emplace_back(output_queue_info);
            if (output_queue_info.queue_id == Common::kDummyQId) {
                continue;
            }
            queue_dev_set_.emplace(output_queue_info.device_id);
            dev_output_queue_map_[output_queue_info.device_id].emplace(output_queue_info.queue_id, output_queue_info.is_proxy_queue);
            if (output_queue_info.is_proxy_queue) {
                with_proxy_queue_ = true;
                continue;
            }
            output_to_flow_func_processor_idx_[output_queue_info.queue_id].emplace_back(func_processors_.size());
        }
        const auto &input_align_attrs = model.GetInputAlignAttrs();
        flow_func_processor->SetInputAlignAttrs(input_align_attrs.align_max_cache_num, input_align_attrs.align_timeout,
            input_align_attrs.drop_when_not_align);
        func_processors_.emplace_back(flow_func_processor);
        UDF_RUN_LOG_INFO("Flow func[instance_name:%s, funcName:%s] IO info: inputQs=%s, outputQs=%s.",
            instance_name.c_str(),
            input_map.first.c_str(),
            ToString(input_map.second).c_str(),
            ToString(outputs_queue_infos).c_str());
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::CheckInputOutputMapsValid(const std::map<std::string, std::vector<uint32_t>> &input_maps,
    uint32_t inputs_num, const std::map<std::string, std::vector<uint32_t>> &output_maps, uint32_t outputs_num) {
    for (auto iter = input_maps.begin(); iter != input_maps.end(); iter++) {
        for (auto index : iter->second) {
            if (index >= inputs_num) {
                UDF_LOG_ERROR("func name[%s]'s input map index[%u] is invalid, valid range is [0, %u).",
                    iter->first.c_str(),
                    index,
                    inputs_num);
                return FLOW_FUNC_FAILED;
            }
        }
    }
    for (auto iter = output_maps.begin(); iter != output_maps.end(); iter++) {
        for (auto index : iter->second) {
            if (index >= outputs_num) {
                UDF_LOG_ERROR("func name[%s]'s output map index[%u] is invalid, valid range is [0, %u).",
                    iter->first.c_str(),
                    index,
                    outputs_num);
                return FLOW_FUNC_FAILED;
            }
        }
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::GetRealQueueInfos(const std::string &flow_func_name,
                                         const std::vector<QueueDevInfo> &input_queue_infos,
                                         const std::map<std::string, std::vector<uint32_t>> &multi_func_input_maps,
                                         std::map<std::string, std::vector<QueueDevInfo>> &real_input_queue_maps) {
    if (multi_func_input_maps.empty()) {
        // for single func, input_maps size is zero
        real_input_queue_maps.emplace(flow_func_name, input_queue_infos);
    } else {
        for (auto &input_maps : multi_func_input_maps) {
            std::vector<QueueDevInfo> queue_infos;
            queue_infos.reserve(input_maps.second.size());
            std::transform(input_maps.second.cbegin(), input_maps.second.cend(), std::back_inserter(queue_infos),
                [&input_queue_infos](uint32_t index) {
                    return input_queue_infos[index];
                });
            real_input_queue_maps.emplace(input_maps.first, std::move(queue_infos));
        }
    }
}

int32_t FlowFuncExecutor::GetModelQueueInfos(const FlowFuncModel &model, ModelQueueInfos &model_queue_infos) {
    if (GlobalConfig::Instance().IsNpuSched()) {
        return npu_sched_processor_->LoadNpuSchedModel(model, model_queue_infos);
    } else {
        model_queue_infos.input_queues = model.GetInputQueues();
        model_queue_infos.output_queues = model.GetOutputQueues();
        model_queue_infos.func_input_maps = model.GetMultiFuncInputMaps();
        model_queue_infos.func_output_maps = model.GetMultiFuncOutputMaps();
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::InitNpuSchedProcessor() {
    if (GlobalConfig::Instance().IsNpuSched()) {
        npu_sched_processor_ = MakeShared<NpuSchedProcessor>();
        if (npu_sched_processor_ == nullptr) {
            UDF_LOG_ERROR("failed to alloc npu sched processor.");
            return FLOW_FUNC_FAILED;
        }
        int32_t init_ret = npu_sched_processor_->Initialize(GlobalConfig::Instance().GetRunningDeviceId());
        if (init_ret != FLOW_FUNC_SUCCESS) {
            (void)SendMessageByResponseQueue(ControlMessageType::kInit, init_ret);
            return init_ret;
        }
        int32_t send_ret = SendMessageByResponseQueue(ControlMessageType::kInit, FLOW_FUNC_SUCCESS);
        if (send_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("failed to send init message.");
            return send_ret;
        }
        Mbuf *control_muff = nullptr;
        // mast wait 30s
        auto deque_ret = request_queue_wrapper_->DequeueWithTimeout(control_muff, 30 * 1000);
        if ((deque_ret != HICAID_SUCCESS)) {
            UDF_LOG_ERROR("Dequeue message and wait notify message failed, ret = %d.", deque_ret);
            return FLOW_FUNC_FAILED;
        }
        auto mbuf_deleter = [control_muff]() { (void)halMbufFree(control_muff); };
        ScopeGuard mbuf_guard(mbuf_deleter);
        ff::deployer::ExecutorRequest req_msg;
        RequestMsgType msg_type = RequestMsgType::kControlMsg;
        auto parse_ret = ParseRequestMessage(control_muff, req_msg, msg_type);
        if (parse_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Parse notify failed, ret = %d.", parse_ret);
            return parse_ret;
        }
        if (msg_type != RequestMsgType::kNotify) {
            UDF_LOG_ERROR("expect notify message=%d but got %d message.", RequestMsgType::kNotify, msg_type);
            return parse_ret;
        }
        UDF_RUN_LOG_INFO("receive deployer notify continue message.");
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::GetUsrInvokedModelKey(const std::string &name_with_scope, const std::string &scope,
                                                std::string &invoke_name) {
    if (scope.empty()) {
        invoke_name = name_with_scope;
        return FLOW_FUNC_SUCCESS;
    }
    if ((name_with_scope.size() >= scope.size()) && (name_with_scope.compare(0, scope.size(), scope) == 0)) {
        invoke_name = name_with_scope.substr(scope.size());
        return FLOW_FUNC_SUCCESS;
    }

    UDF_LOG_ERROR("Get the usr invoked model key failed, dataFlowScope=%s, invoked model key=%s.",
                  scope.c_str(), name_with_scope.c_str());
    return FLOW_FUNC_FAILED;
}

int32_t FlowFuncExecutor::CreateFlowModel(const std::shared_ptr<FlowFuncParams> &params, const FlowFuncModel &model) {
    const auto &name = model.GetName();
    const auto &flow_func_name = model.GetFlowFuncName();
    const auto &invokedModelQueueInfos = model.GetInvokedModelQueueInfos();
    const auto &input_align_attrs = model.GetInputAlignAttrs();
    for (const auto &invoked_model_queue_info : invokedModelQueueInfos) {
        std::string invoke_name;
        if (GetUsrInvokedModelKey(invoked_model_queue_info.first, model.GetDataFlowScope(), invoke_name) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("GetUsrInvokedModelKey failed, name=%s, flow_func_name=%s, invoked model key=%s.",
                name.c_str(), flow_func_name.c_str(), invoked_model_queue_info.first.c_str());
            return FLOW_FUNC_FAILED;
        }
        std::unique_ptr<DataAligner> dataAligner;
        if (input_align_attrs.align_max_cache_num > 0) {
            dataAligner.reset(new(std::nothrow) DataAligner(invoked_model_queue_info.second.fetch_queue_infos.size(),
                input_align_attrs.align_max_cache_num,
                input_align_attrs.align_timeout, input_align_attrs.drop_when_not_align));
            if (dataAligner == nullptr) {
                UDF_LOG_ERROR("alloc DataAligner failed, name=%s, flow_func_name=%s.", name.c_str(),
                    flow_func_name.c_str());
                return FLOW_FUNC_FAILED;
            }
        }
        std::unique_ptr<FlowModel> impl(
            new(std::nothrow) FlowModelImpl(invoked_model_queue_info.second.feed_queue_infos,
                invoked_model_queue_info.second.fetch_queue_infos, std::move(dataAligner)));
        if (impl == nullptr) {
            UDF_LOG_ERROR("alloc FlowModelImpl failed, name=%s, flow_func_name=%s, invoked model key=%s.",
                name.c_str(), flow_func_name.c_str(), invoke_name.c_str());
            return FLOW_FUNC_FAILED;
        }
        params->AddFlowModel(invoke_name, std::move(impl));
        UDF_RUN_LOG_INFO("Flow func(ppName:%s) invoke model IO info: invoked key=%s, feedQs=%s, fetchQs=%s.",
            name.c_str(), invoke_name.c_str(),
            ToString(invoked_model_queue_info.second.feed_queue_infos).c_str(),
            ToString(invoked_model_queue_info.second.fetch_queue_infos).c_str());
        for (const auto &feed_queue_info : invoked_model_queue_info.second.feed_queue_infos) {
            with_proxy_queue_ = with_proxy_queue_ || feed_queue_info.is_proxy_queue;
            queue_dev_set_.emplace(feed_queue_info.device_id);
        }
        for (const auto &fetch_queue_info : invoked_model_queue_info.second.fetch_queue_infos) {
            with_proxy_queue_ = with_proxy_queue_ || fetch_queue_info.is_proxy_queue;
            queue_dev_set_.emplace(fetch_queue_info.device_id);
        }
    }

    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::Init(const std::vector<std::unique_ptr<FlowFuncModel>> &models) {
    UDF_RUN_LOG_INFO("Init start, model num=%zu.", models.size());
    int32_t init_ret = InitMessageQueue();
    if (init_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to init message queue.");
        return init_ret;
    }
    init_ret = InitNpuSchedProcessor();
    if (init_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to init npu sched processor.");
        return init_ret;
    }

    async_executor_ = MakeShared<AsyncExecutor>("udf_async_", kAsyncExecutorThreadNum);
    bool is_cpu_num_from_attr = false;
    for (size_t model_idx = 0UL; model_idx < models.size(); ++model_idx) {
        const auto &model = models[model_idx];
        const auto &name = model->GetName();
        const auto &instance_name = model->GetModelInstanceName();
        const auto &flow_func_name = model->GetFlowFuncName();
        ModelQueueInfos model_queue_infos = {};
        if (GetModelQueueInfos(*model, model_queue_infos) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("failed to get model[%s] queue infos.", instance_name.c_str());
            return FLOW_FUNC_FAILED;
        }
        const auto &input_queue_infos = model_queue_infos.input_queues;
        const auto &output_queue_infos = model_queue_infos.output_queues;
        const auto &multi_func_input_maps = model_queue_infos.func_input_maps;
        const auto &multi_func_output_maps = model_queue_infos.func_output_maps;
        const auto &stream_input_func_names = model->GetStreamInputFuncNames();

        if (CheckInputOutputMapsValid(multi_func_input_maps, input_queue_infos.size(), multi_func_output_maps,
            output_queue_infos.size()) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("instance name[%s]'s func inputs/outputs map is invalid.", instance_name.c_str());
            return FLOW_FUNC_FAILED;
        }

        int32_t running_device_id = 0;
        if (GlobalConfig::Instance().IsOnDevice()) {
            running_device_id = static_cast<int32_t>(GlobalConfig::Instance().GetDeviceId());
        } else {
            running_device_id = GlobalConfig::Instance().GetRunningDeviceId();
        }

        // create flow func params
        std::shared_ptr<FlowFuncParams> params = MakeShared<FlowFuncParams>(name, input_queue_infos.size(),
            output_queue_infos.size(), running_device_id, GlobalConfig::Instance().GetDeviceId());
        if (params == nullptr) {
            UDF_LOG_ERROR(
                "Failed to create params, name[%s], output queue size[%zu].", name.c_str(), output_queue_infos.size());
            return FLOW_FUNC_FAILED;
        }
        params->SetLibPath(model->GetLibPath());
        params->SetAttrMap(model->GetNodeAttrMap());
        params->SetWorkPath(model->GetWorkPath());
        params->SetStreamInputFuncNames(stream_input_func_names);
        params->SetModelUuid(model->GetModelUuid());
        params->SetHeadInfo(model->IsHead());
        params->SetInstanceName(instance_name);
        params->SetScope(model->GetScope());
        params->SetRunnningInstanceId(model->GetReplicaIdx());
        params->SetRunnningInstanceNum(model->GetReplicaNum());
        params->SetInvokedScopes(model->GetInvokedScopes());
        if (!GlobalConfig::Instance().IsNpuSched()) {
            const QueueDevInfo &status_output_queue = model->GetStatusOutputQueue();
            params->SetStatusOutputQueue(status_output_queue);
            const bool need_report_status = model->NeedReportStatus();
            params->SetNeedReportStatusFlag(need_report_status);
            params->SetEnableRaiseException(model->GetEnableRaiseException());
            if (need_report_status || model->GetEnableRaiseException()) {
                status_output_queue_map_[status_output_queue.device_id].emplace_back(status_output_queue.queue_id);
                with_proxy_queue_ = with_proxy_queue_ || status_output_queue.is_proxy_queue;
                queue_dev_set_.emplace(status_output_queue.device_id);
            }
        }

        if (CreateFlowModel(params, *model) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("create flow model failed, name=%s, flow_func_name=%s.", name.c_str(), flow_func_name.c_str());
            return FLOW_FUNC_FAILED;
        }

        std::map<std::string, std::vector<QueueDevInfo>> real_input_queue_maps;
        GetRealQueueInfos(flow_func_name, input_queue_infos, multi_func_input_maps, real_input_queue_maps);
        if (CreateFlowFuncProcessor(*model, real_input_queue_maps, multi_func_output_maps, output_queue_infos, params) !=
            FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR(
                "create multi func processor failed, name=%s, flow_func_name=%s", name.c_str(), flow_func_name.c_str());
            return FLOW_FUNC_FAILED;
        }
        func_params_.emplace_back(params);

        UpdatePriority(model->GetModelEschedProcessPriority(), esched_process_priority_);
        UpdatePriority(model->GetModelEschedEventPriority(), esched_event_priority_);
        if (!GlobalConfig::Instance().IsOnDevice()) {
            uint32_t cpu_num = 1U;
            bool is_attr_get = false;
            if (model->GetCpuNumFromAttr(cpu_num, is_attr_get) != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("get cpu num from attr fail, name=%s, flow_func_name=%s.",
                    name.c_str(), flow_func_name.c_str());
                return FLOW_FUNC_FAILED;
            }
            if (is_attr_get) {
                is_cpu_num_from_attr = true;
                // add one extra thread for handling events
                cpu_num_ = (cpu_num > cpu_num_ ? cpu_num : cpu_num_) + 1U;
            }
        }
        UDF_RUN_LOG_INFO("[UdfModelEschedPriority]flow func init end, name=%s, flow_func_name=%s, idx=%zu, "
                         "process priority=%d, event priority=%d.", name.c_str(), flow_func_name.c_str(), model_idx,
            esched_process_priority_, esched_event_priority_);
    }
    if (!is_cpu_num_from_attr) {
        cpu_num_ = static_cast<uint32_t>(func_processors_.size() + 1U);
    }
    UDF_LOG_INFO("cpu num is %u.", cpu_num_);
    const auto drv_ret = InitDrv();
    if (drv_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Init driver failed.");
        return drv_ret;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::Destroy() {
    UnsubscribeInputQueue();
    UnsubscribeOutputQueue();
    input_to_flow_func_processor_idx_.clear();
    output_to_flow_func_processor_idx_.clear();
    func_processors_.clear();
    if (npu_sched_processor_ != nullptr) {
        npu_sched_processor_->Finalize();
    }
    FlowFuncManager::Instance().Reset();
}

int32_t FlowFuncExecutor::SetExecutorEschedPriority() const {
    uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    if (esched_process_priority_ != kUserUnsetESchedPriority) {
        const drvError_t drv_ret = halEschedSetPidPriority(device_id,
            static_cast<SCHEDULE_PRIORITY>(esched_process_priority_));
        if (drv_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("Failed to set pid priority, device_id=%u, priority=%d, drv_ret=%d.",
                device_id, esched_process_priority_, static_cast<int32_t>(drv_ret));
            return FLOW_FUNC_ERR_DRV_ERROR;
        }
        UDF_LOG_INFO("[UdfModelEschedPriority] Succeed to set eshced process priority=%d.",
            esched_process_priority_);
    }
    if (esched_event_priority_ != kUserUnsetESchedPriority) {
        auto drv_ret = halEschedSetEventPriority(device_id, EVENT_QUEUE_EMPTY_TO_NOT_EMPTY,
            static_cast<SCHEDULE_PRIORITY>(esched_event_priority_));
        if (drv_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("Failed to set event[%d] priority, device_id=%u, priority=%d, drv_ret=%d.",
                static_cast<int32_t>(EVENT_QUEUE_EMPTY_TO_NOT_EMPTY), device_id, esched_event_priority_,
                static_cast<int32_t>(drv_ret));
            return FLOW_FUNC_ERR_DRV_ERROR;
        }
        drv_ret = halEschedSetEventPriority(device_id, EVENT_QUEUE_FULL_TO_NOT_FULL,
            static_cast<SCHEDULE_PRIORITY>(esched_event_priority_));
        if (drv_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("Failed to set event[%d] priority, device_id=%u, priority=%d, drv_ret=%d.",
                static_cast<int32_t>(EVENT_QUEUE_FULL_TO_NOT_FULL), device_id, esched_event_priority_,
                static_cast<int32_t>(drv_ret));
            return FLOW_FUNC_ERR_DRV_ERROR;
        }
        UDF_LOG_INFO("[UdfModelEschedPriority] Succeed to set eshced event priority=%d.",
            esched_event_priority_);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::InitQueue() const {
    if (with_proxy_queue_) {
        // wait time out is 60s
        constexpr uint32_t kWaitTimeout = 60 * 1000;
        (void)FlowFuncDrvManager::Instance().WaitBindHostPid(kWaitTimeout);
    }

    for (uint32_t queue_device_id : queue_dev_set_) {
        drvError_t drv_ret = halQueueInit(queue_device_id);
        if ((drv_ret != DRV_ERROR_NONE) && (drv_ret != DRV_ERROR_REPEATED_INIT)) {
            UDF_LOG_ERROR("halQueueInit error, queue_device_id=%u, drv_ret=%d", queue_device_id,
                static_cast<int32_t>(drv_ret));
            return FLOW_FUNC_ERR_QUEUE_ERROR;
        }
        UDF_LOG_INFO("halQueueInit success, queue_device_id=%u", queue_device_id);
    }
    return FLOW_FUNC_SUCCESS;
}


int32_t FlowFuncExecutor::InitDrv() const {
    UDF_RUN_LOG_INFO("ready to init drv, withProxyQueue=%d", static_cast<int32_t>(with_proxy_queue_));
    int32_t queue_init_ret = InitQueue();
    if (queue_init_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("init queue failed, ret=%d.", queue_init_ret);
        return queue_init_ret;
    }

    auto ret = SetExecutorEschedPriority();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to set esched priority, ret=%d.", ret);
        return ret;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::InitMessageQueue() {
    const uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    const uint32_t req_queue_id = GlobalConfig::Instance().GetReqQueueId();
    const uint32_t rsp_queue_id = GlobalConfig::Instance().GetRspQueueId();
    if ((req_queue_id == UINT32_MAX) || (rsp_queue_id == UINT32_MAX)) {
        UDF_LOG_INFO("Message queues are not created.");
        return FLOW_FUNC_SUCCESS;
    }
    // 1. construct wrapper for request queue and response queue
    try {
        request_queue_wrapper_.reset(new(std::nothrow) QueueWrapper(device_id, req_queue_id));
        response_queue_wrapper_.reset(new(std::nothrow) QueueWrapper(device_id, rsp_queue_id));
        if ((request_queue_wrapper_ == nullptr) || (response_queue_wrapper_ == nullptr)) {
            UDF_LOG_ERROR("Make request/response queue wrapper failed.");
            return FLOW_FUNC_FAILED;
        }
    } catch (std::exception &e) {
        UDF_LOG_ERROR("Init message queue failed, error=%s.", e.what());
        return FLOW_FUNC_FAILED;
    }

    // 2. attach queue
    auto drv_ret = halQueueInit(device_id);
    if ((drv_ret != DRV_ERROR_NONE) && (drv_ret != DRV_ERROR_REPEATED_INIT)) {
        UDF_LOG_ERROR("halQueueInit error, device_id=%u, drv_ret=%d", device_id, static_cast<int32_t>(drv_ret));
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }
    UDF_LOG_INFO("halQueueInit success, queue_device_id=%u", device_id);
    drv_ret = halQueueAttach(device_id, req_queue_id, kAttachWaitTimeout);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("attached request queue[%u] failed, ret[%d], queue_device_id[%u]", req_queue_id,
                      static_cast<int32_t>(drv_ret), device_id);
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }

    drv_ret = halQueueAttach(device_id, rsp_queue_id, kAttachWaitTimeout);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("attached response queue[%u] failed, ret[%d], queue_device_id[%u]", rsp_queue_id,
                      static_cast<int32_t>(drv_ret), device_id);
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }
    UDF_LOG_INFO("Attach message %u %u success.", req_queue_id, rsp_queue_id);
    QueueSetInputPara input_param;
    QueueSetInput input;
    input.queSetWorkMode.qid = req_queue_id;
    input.queSetWorkMode.workMode = static_cast<uint32_t>(QUEUE_MODE_PULL);
    input_param.inBuff = static_cast<void *>(&input);
    input_param.inLen = static_cast<uint32_t>(sizeof(QueueSetInput));
    (void)halQueueSet(device_id, QUEUE_SET_WORK_MODE, &input_param);
    drv_ret = halQueueSubscribe(device_id, req_queue_id, GlobalConfig::Instance().GetMainSchedGroupId(),
                               static_cast<int32_t>(QUEUE_TYPE_SINGLE));
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Failed to subscribe event for queue[%u], ret[%d].", req_queue_id, static_cast<int32_t>(drv_ret));
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}


int32_t FlowFuncExecutor::SendSwitchSoftModeEvent() const {
    event_summary event_info_summary = {};
    char msg_value[8] = {}; // 2-byte alignment and less than 36 bytes
    event_info_summary.pid = getpid();
    event_info_summary.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdSwitchToSoftSchedMode);
    event_info_summary.subevent_id = 0U;
    event_info_summary.msg = msg_value;
    event_info_summary.msg_len = static_cast<uint32_t>(sizeof(msg_value)); // soft mode event msg len must != 0
    event_info_summary.dst_engine = GlobalConfig::Instance().IsRunOnAiCpu() ? ACPU_LOCAL : CCPU_LOCAL;
    event_info_summary.grp_id = GlobalConfig::Instance().GetMainSchedGroupId();
    event_info_summary.tid = 0U;

    drvError_t ret = halEschedSubmitEventToThread(GlobalConfig::Instance().GetDeviceId(), &event_info_summary);
    if (ret != DRV_ERROR_NONE) {
        UDF_RUN_LOG_INFO("Submit switch soft sched mode get ret=%d.", static_cast<int32_t>(ret));
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    UDF_RUN_LOG_INFO("Submit switch soft sched mode succ.");
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::SendProcessorInitEvent() const {
    event_summary event_info_summary = {};
    event_info_summary.pid = getpid();
    event_info_summary.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdProcessorInit);
    event_info_summary.subevent_id = 0U;
    event_info_summary.msg = nullptr;
    event_info_summary.msg_len = 0U;
    event_info_summary.dst_engine = GlobalConfig::Instance().IsRunOnAiCpu() ? ACPU_LOCAL : CCPU_LOCAL;
    event_info_summary.grp_id = GlobalConfig::Instance().GetMainSchedGroupId();
    drvError_t ret = DRV_ERROR_NONE;
    const int32_t submit_event_retry_num = 3;
    int32_t retry_num = 0;
    do {
        ret = halEschedSubmitEvent(GlobalConfig::Instance().GetDeviceId(), &event_info_summary);
        if (ret != DRV_ERROR_NONE) {
            retry_num++;
            UDF_RUN_LOG_INFO("Submitting flow func processor to init will retry 100ms later, current retry num %d.",
                retry_num);
            // retry every 100ms
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        UDF_RUN_LOG_INFO("Submit flow func processor init event success.");
        return FLOW_FUNC_SUCCESS;
    } while (retry_num < submit_event_retry_num);
    UDF_LOG_ERROR("Failed to submit flow func processor init event. drv_ret=%d.", static_cast<int32_t>(ret));
    return FLOW_FUNC_ERR_DRV_ERROR;
}

int32_t FlowFuncExecutor::Start() {
    running_ = true;
    // as timer need send event, so default add 1 to default thread num
    auto ret = udf_thread_pool_.Init(
        [this](uint32_t thread_idx) { ThreadLoop(thread_idx); }, cpu_num_);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("thread pool init failed, ret=%d.", ret);
        Stop();
        return ret;
    }
    // on host, remain 1 extra thread for handling event
    auto worker_num =
        GlobalConfig::Instance().IsRunOnAiCpu() ? udf_thread_pool_.GetThreadNum() : (udf_thread_pool_.GetThreadNum() - 1);
    GlobalConfig::Instance().SetWorkerNum(worker_num);

    udf_thread_pool_.WaitAllThreadReady();
    if (!running_) {
        UDF_LOG_ERROR("thread start failed.");
        return FLOW_FUNC_FAILED;
    }

    ret = FlowFuncManager::Instance().Init();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Flow func manager init failed, ret=%d.", ret);
        Stop();
        return ret;
    }

    // start timeout thread.
    FlowFuncTimer::Instance().Init(GlobalConfig::Instance().GetDeviceId());
    ret = FlowFuncLoggerManager::Instance().Init();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Flow func logger manager init failed, ret=%d.", ret);
        Stop();
        return ret;
    }

    ret = SendProcessorInitEvent();
    if (ret != FLOW_FUNC_SUCCESS) {
        Stop();
        return ret;
    }
    (void)SendSwitchSoftModeEvent(); // try switch driver soft sched mode

    statistic_timer_handle_ = FlowFuncTimer::Instance().CreateTimer([this]() {
        DumpMetrics();
    });
    // period is 80s, as running log limit 50 logs per hour.
    constexpr uint32_t kStatisticTimerPeriod = 80 * 1000;
    (void)FlowFuncTimer::Instance().StartTimer(statistic_timer_handle_, kStatisticTimerPeriod, false);
    if (GlobalConfig::Instance().IsOnDevice()) {
        MonitorParentExit();
    }
    MonitorTermSignal();
    MemoryStatisticManager::Instance().Init(GlobalConfig::Instance().GetMemGroupName());
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::MonitorParentExit() {
    auto start_parent_pid = getppid();
    const auto monitor_func = [this, start_parent_pid]() {
        auto current_parent_pid = getppid();
        // if start default when parent exit, parent will become 1 or -1.
        if ((current_parent_pid != start_parent_pid) && (start_parent_pid != 1)) {
            static int32_t stop_times = 0;
            // wait 5 times to normal stop.
            if (stop_times < 5) {
                UDF_RUN_LOG_INFO("parent pid[%d] exit, current_parent_pid=%d, udf will stop, times=%d.", start_parent_pid,
                    current_parent_pid, stop_times);
                Stop();
                ++stop_times;
            } else {
                UDF_RUN_LOG_INFO("parent pid[%d] exit, but udf can not stop normally, so kill itself.", start_parent_pid);
                (void)kill(getpid(), SIGKILL);
            }
        }
    };
    // invokeByWorker must be false, as when shutdown, worker will be
    monitor_parent_timer_handle_ = FlowFuncTimer::Instance().CreateTimer(monitor_func, false);
    // monitor parent period is 1s.
    constexpr uint32_t monitor_parent_timer_period = 1 * 1000;
    (void)FlowFuncTimer::Instance().StartTimer(monitor_parent_timer_handle_, monitor_parent_timer_period, false);
}

void FlowFuncExecutor::MonitorTermSignal() {
    const auto monitor_func = [this]() {
        if (!recv_term_signal_) {
            return;
        }
        recv_term_signal_ = false;
        event_summary event_info_summary = {};
        event_info_summary.pid = getpid();
        event_info_summary.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdNotifyThreadExit);
        event_info_summary.subevent_id = 0;
        event_info_summary.msg = nullptr;
        event_info_summary.msg_len = 0U;
        event_info_summary.dst_engine = GlobalConfig::Instance().IsRunOnAiCpu() ? ACPU_LOCAL : CCPU_LOCAL;
        uint32_t thread_num = udf_thread_pool_.GetThreadNum();
        for (uint32_t i = 0; i < thread_num; ++i) {
            uint32_t main_sched_group_id = GlobalConfig::Instance().GetMainSchedGroupId();
            uint32_t worker_sched_group_id = GlobalConfig::Instance().GetWorkerSchedGroupId();
            event_info_summary.grp_id = (i == (thread_num - 1) ? main_sched_group_id : worker_sched_group_id);
            drvError_t ret = halEschedSubmitEvent(GlobalConfig::Instance().GetDeviceId(), &event_info_summary);
            if (ret != DRV_ERROR_NONE) {
                UDF_LOG_WARN("Failed to submit notify thread exit event, drv_ret=%d.", static_cast<int32_t>(ret));
            }
        }
        UDF_RUN_LOG_INFO("receive term signal, notify all thread exit end, thread_num=%u.", thread_num);
    };
    // invokeByWorker must be false, as when shutdown, worker will be
    monitor_term_signal_timer_handle_ = FlowFuncTimer::Instance().CreateTimer(monitor_func, false);
    // monitor period is 10ms.
    constexpr uint32_t kMonitorTermSignalTimerPeriod = 10;
    (void)FlowFuncTimer::Instance().StartTimer(monitor_term_signal_timer_handle_, kMonitorTermSignalTimerPeriod, false);
}

void FlowFuncExecutor::CheckReplenishSchedule() {
    for (size_t idx = 0UL; idx < func_processors_.size(); ++idx) {
        if (func_processors_[idx]->NeedReplenishSchedule()) {
            UDF_LOG_INFO("processor[%s] need replenish schedule event",
                func_processors_[idx]->GetFlowFuncInfo().c_str());
            (void)ScheduleFlowFunc(idx);
        }
    }
}

int32_t FlowFuncExecutor::ScheduleFlowFunc(size_t flow_func_processor_idx) const {
    UDF_LOG_DEBUG("schedule flow func, flow_func_processor_idx=%zu.", flow_func_processor_idx);
    int32_t ret = SubmitEvent(GlobalConfig::Instance().GetWorkerSchedGroupId(), UdfEvent::kEventIdFlowFuncExecute,
        static_cast<uint32_t>(flow_func_processor_idx));
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to submit flow func execute event. flow_func_processor_idx=%zu.", flow_func_processor_idx);
        return ret;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::SubmitEvent(uint32_t group_id, uint32_t event_id, uint32_t sub_event_id) {
    UDF_LOG_DEBUG("submit event, group_id=%u, event_id=%u, sub_event_id=%u.", group_id, event_id, sub_event_id);
    event_summary event_info_summary = {};
    event_info_summary.pid = getpid();
    event_info_summary.event_id = static_cast<EVENT_ID>(event_id);
    event_info_summary.subevent_id = sub_event_id;
    event_info_summary.msg = nullptr;
    event_info_summary.msg_len = 0U;
    event_info_summary.dst_engine = GlobalConfig::Instance().IsRunOnAiCpu() ? ACPU_LOCAL : CCPU_LOCAL;
    event_info_summary.grp_id = group_id;

    drvError_t ret = halEschedSubmitEvent(GlobalConfig::Instance().GetDeviceId(), &event_info_summary);
    if (ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Failed to submit event, drv_ret=%d, group_id=%u, event_id=%u, sub_event_id=%u.",
            static_cast<int32_t>(ret), group_id, event_id, sub_event_id);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::Stop(bool recv_term_signal) {
    running_ = false;
    GlobalConfig::Instance().SetExitFlag(true);
    if (recv_term_signal) {
        recv_term_signal_ = true;
    }
}

void FlowFuncExecutor::WaitForStop() {
    udf_thread_pool_.WaitForStop();
    FlowFuncLoggerManager::Instance().Finalize();

    if (statistic_timer_handle_ != nullptr) {
        (void)FlowFuncTimer::Instance().StopTimer(statistic_timer_handle_);
        (void)FlowFuncTimer::Instance().DeleteTimer(statistic_timer_handle_);
        statistic_timer_handle_ = nullptr;
    }
    if (monitor_term_signal_timer_handle_ != nullptr) {
        (void)FlowFuncTimer::Instance().StopTimer(monitor_term_signal_timer_handle_);
        (void)FlowFuncTimer::Instance().DeleteTimer(monitor_term_signal_timer_handle_);
        monitor_term_signal_timer_handle_ = nullptr;
    }

    if (monitor_parent_timer_handle_ != nullptr) {
        (void)FlowFuncTimer::Instance().StopTimer(monitor_parent_timer_handle_);
        (void)FlowFuncTimer::Instance().DeleteTimer(monitor_parent_timer_handle_);
        monitor_parent_timer_handle_ = nullptr;
    }

    FlowFuncTimer::Instance().Finalize();
    DumpMetrics(true);
    MemoryStatisticManager::Instance().Finalize();
}

int32_t FlowFuncExecutor::SubscribeInvokeModelEvent(uint32_t thread_idx) const {
    uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    uint32_t invoke_model_sched_group_id = GlobalConfig::Instance().GetInvokeModelSchedGroupId();
    drvError_t ret = halEschedSubscribeEvent(device_id, invoke_model_sched_group_id, thread_idx,
        kInvokeModelWaitEventMask);
    if (ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR(
            "halEschedSubscribeEvent failed, device_id[%u], group_id[%u], thread_idx[%u] eventBitmap[%lu].",
            device_id, invoke_model_sched_group_id, thread_idx, kInvokeModelWaitEventMask);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    struct event_info event = {};
    ret = halEschedWaitEvent(device_id, invoke_model_sched_group_id, thread_idx, 0, &event);
    if (ret != DRV_ERROR_NO_EVENT) {
        UDF_LOG_ERROR(
            "halEschedWaitEvent ret=%d must be no event=%d, device_id[%u], group_id[%u], thread_idx[%u] eventBitmap[%lu].",
            static_cast<int32_t>(ret), static_cast<int32_t>(DRV_ERROR_NO_EVENT), device_id, invoke_model_sched_group_id,
            thread_idx,
            kInvokeModelWaitEventMask);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::SubscribeFlowMsgQueueEvent(uint32_t thread_idx) const {
    uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    uint32_t flow_msg_queue_sched_group_id = GlobalConfig::Instance().GetFlowMsgQueueSchedGroupId();
    drvError_t ret = halEschedSubscribeEvent(device_id, flow_msg_queue_sched_group_id, thread_idx,
        kFlowMsgQueueWaitEventMask);
    if (ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR(
            "halEschedSubscribeEvent failed, device_id[%u], group_id[%u], thread_idx[%u] eventBitmap[%lu].",
            device_id, flow_msg_queue_sched_group_id, thread_idx, kFlowMsgQueueWaitEventMask);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    struct event_info event = {};
    ret = halEschedWaitEvent(device_id, flow_msg_queue_sched_group_id, thread_idx, 0, &event);
    if (ret != DRV_ERROR_NO_EVENT) {
        UDF_LOG_ERROR(
            "halEschedWaitEvent ret=%d must be no event=%d, device_id[%u], group_id[%u], thread_idx[%u] eventBitmap[%lu].",
            static_cast<int32_t>(ret), static_cast<int32_t>(DRV_ERROR_NO_EVENT), device_id, flow_msg_queue_sched_group_id,
            thread_idx,
            kFlowMsgQueueWaitEventMask);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::ThreadLoop(uint32_t thread_idx) {
    GlobalConfig::Instance().SetCurrentSchedThreadIdx(thread_idx);
    if (!GlobalConfig::Instance().IsLimitRunBuiltinUdf()) {
        int32_t thread_ret = FlowFuncThreadPool::ThreadSecureCompute();
        if (thread_ret != FLOW_FUNC_SUCCESS) {
            Stop();
            udf_thread_pool_.ThreadAbnormal(thread_idx);
            UDF_LOG_ERROR("ThreadSecureCompute failed, thread_idx[%u].", thread_idx);
            return;
        }
    }

    int32_t ret = SubscribeInvokeModelEvent(thread_idx);
    if (ret != FLOW_FUNC_SUCCESS) {
        Stop();
        udf_thread_pool_.ThreadAbnormal(thread_idx);
        UDF_LOG_ERROR("SubscribeInvokeModelEvent failed, thread_idx[%u].", thread_idx);
        return;
    }

    ret = SubscribeFlowMsgQueueEvent(thread_idx);
    if (ret != FLOW_FUNC_SUCCESS) {
        Stop();
        udf_thread_pool_.ThreadAbnormal(thread_idx);
        UDF_LOG_ERROR("SubscribeFlowMsgQueueEvent failed, thread_idx[%u].", thread_idx);
        return;
    }

    uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    uint32_t main_sched_group_id = GlobalConfig::Instance().GetMainSchedGroupId();
    uint32_t worker_sched_group_id = GlobalConfig::Instance().GetWorkerSchedGroupId();
    uint32_t sched_group_id = (thread_idx == (cpu_num_ - 1) ? main_sched_group_id : worker_sched_group_id);
    uint64_t wait_event_mask = (thread_idx == (cpu_num_ - 1) ? kWaitEventMask : kWorkerWaitEventMask);
    if (GlobalConfig::Instance().IsOnDevice()) {
        wait_event_mask = (kWaitEventMask | kWorkerWaitEventMask);
    }
    drvError_t drv_ret = halEschedSubscribeEvent(device_id, sched_group_id, thread_idx, wait_event_mask);
    if (drv_ret != DRV_ERROR_NONE) {
        Stop();
        udf_thread_pool_.ThreadAbnormal(thread_idx);
        UDF_LOG_ERROR("halEschedSubscribeEvent failed, device_id[%u], group_id[%u], thread_idx[%u] eventBitmap[%lu].",
            device_id, sched_group_id, thread_idx, wait_event_mask);
        return;
    }
    UDF_RUN_LOG_INFO("thread[%u] subscribe event end, sched_group_id=%u, eventBitmap=%lu.",
        thread_idx, sched_group_id, wait_event_mask);
    udf_thread_pool_.ThreadReady(thread_idx);

    struct event_info event = {};
    // default wait timeout 2s
    constexpr int32_t kWaitTimeout = 2000;
    uint32_t timeout_times = 0U;
    while (running_) {
        drvError_t sched_ret = halEschedWaitEvent(device_id, sched_group_id, thread_idx, kWaitTimeout, &event);
        if (sched_ret == DRV_ERROR_NONE) {
            GlobalConfig::Instance().SetCurrentSchedGroupId(sched_group_id);
            ProcessEvent(event, thread_idx);
            // reset timeout times
            timeout_times = 0U;
        } else if (sched_ret == DRV_ERROR_SCHED_WAIT_TIMEOUT) {
            // first timeout write log, then continuous timeout 10 times print log once.
            if ((timeout_times % 10U) == 0U) {
                UDF_LOG_DEBUG("wait event timeout,thread index=%u, continuous timeout times=%u.",
                    thread_idx, timeout_times);
            }
            ++timeout_times;
            if (thread_idx == (cpu_num_ - 1)) {
                CheckReplenishSchedule();
            }
        } else {
            // LOG ERROR
            UDF_LOG_ERROR("wait event failed, device_id=%u, threadIndex=%u, group_id=%u, sched_ret=%d.",
                device_id, thread_idx, sched_group_id, static_cast<int32_t>(sched_ret));
        }
    }

    UDF_RUN_LOG_INFO("flow func thread[%u] exit.", thread_idx);
}

void FlowFuncExecutor::ProcessEvent(const struct event_info &event, uint32_t thread_idx) {
    auto event_id = static_cast<uint32_t>(event.comm.event_id);
    const auto proc_func_iter = event_proc_func_map_.find(event_id);
    if (proc_func_iter == event_proc_func_map_.cend()) {
        UDF_LOG_ERROR("no proc func found, event_id=%u, thread_idx=%u", event_id, thread_idx);
        return;
    }
    auto event_proc_func = proc_func_iter->second;
    (this->*(event_proc_func))(event, thread_idx);
}

void FlowFuncExecutor::ProcessProcessorInitEvent(const struct event_info &event, uint32_t thread_idx) {
    (void)event;
    uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    int32_t ret;

    UDF_RUN_LOG_INFO("process processor init event start, thread_idx=%u.", thread_idx);
    // first flow func init.
    for (auto &flow_funcs_processor : func_processors_) {
        ret = flow_funcs_processor->Init(device_id);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("flow_func_processor init failed, flowFuncInfo=%s, ret=%d.",
                flow_funcs_processor->GetFlowFuncInfo().c_str(),
                ret);
            Stop();
            return;
        }
    }

    UDF_RUN_LOG_INFO("start to subscribe output queues.");
    // subscribe queue.
    ret = SubscribeOutputQueue();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("SubscribeOutputQueue failed, ret=%d", ret);
        Stop();
        return;
    }

    UDF_RUN_LOG_INFO("start to subscribe input queues.");
    ret = SubscribeInputQueue();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("SubscribeInputQueue failed, ret=%d", ret);
        Stop();
        return;
    }
    UDF_RUN_LOG_INFO("start to subscribe status output queues.");
    ret = SubscribeStatusOutputQueue();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("SubscribeStatusOutputQueue failed, ret=%d", ret);
        Stop();
        return;
    }
    ret = ScheduleFlowFuncInit();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("schedule flow func init failed, ret=%d", ret);
        Stop();
        return;
    }
    UDF_RUN_LOG_INFO("Process processor init event end.");
}

int32_t FlowFuncExecutor::ScheduleFlowFuncInit() const {
    UDF_RUN_LOG_INFO("start to submit init FlowFunc event.");
    int32_t ret = SubmitEvent(GlobalConfig::Instance().GetMainSchedGroupId(), UdfEvent::kEventIdFlowFuncInit, 0);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to submit flow func init event.");
        return ret;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::ProcessSingleFlowFuncInitEvent(const struct event_info &event, uint32_t thread_idx) {
    auto flow_func_processor_idx = static_cast<size_t>(event.comm.subevent_id);

    UDF_LOG_DEBUG("Flow func single flow func init event begin, flow_func_processor_idx=%zu, thread_idx=%u.",
        flow_func_processor_idx, thread_idx);
    if (flow_func_processor_idx >= func_processors_.size()) {
        UDF_LOG_ERROR("FlowFuncExecute event invalid, flow_func_processor_idx=%zu.", flow_func_processor_idx);
        return;
    }
    // Get flow func processor
    auto flow_func_processor = func_processors_[flow_func_processor_idx];
    int32_t ret = flow_func_processor->InitFlowFunc();
    if (ret == FLOW_FUNC_ERR_INIT_AGAIN) {
        ret = SubmitEvent(GlobalConfig::Instance().GetMainSchedGroupId(),
            UdfEvent::kEventIdSingleFlowFuncInit, event.comm.subevent_id);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Failed to submit single flow func init event, flow_func_processor_idx=%zu",
                flow_func_processor_idx);
        }
        return;
    }
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("flow_func_processor init flow func failed, flowFuncInfo=%s, ret=%d",
            flow_func_processor->GetFlowFuncInfo().c_str(),
            ret);
        Stop();
        return;
    }

    UDF_RUN_LOG_INFO("Start to schedule FlowFunc.");
    ret = ScheduleFlowFunc(flow_func_processor_idx);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("ScheduleFlowFunc failed, ret=%d, idx=%zu, flowFuncInfo=%s.",
            ret,
            flow_func_processor_idx,
            func_processors_[flow_func_processor_idx]->GetFlowFuncInfo().c_str());
        Stop();
        return;
    }

    UDF_RUN_LOG_INFO("Single flow func init event process end");
}

void FlowFuncExecutor::ProcessFlowFuncInitEvent(const struct event_info &event, uint32_t thread_idx) {
    (void)event;
    UDF_RUN_LOG_INFO("FlowFunc init event start, thread_idx=%u.", thread_idx);
    uint32_t need_re_init_num = 0;
    // flow func init.
    for (auto &flow_funcs_processor : func_processors_) {
        int32_t ret = flow_funcs_processor->InitFlowFunc();
        if (ret == FLOW_FUNC_ERR_INIT_AGAIN) {
            ++need_re_init_num;
            continue;
        }
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("flow_func_processor init flow func failed, flowFuncInfo=%s, ret=%d",
                flow_funcs_processor->GetFlowFuncInfo().c_str(),
                ret);
            Stop();
            return;
        }
    }

    if (need_re_init_num > 0) {
        UDF_LOG_INFO("flow func need re init, need_re_init_num=%u.", need_re_init_num);
        int32_t ret = ScheduleFlowFuncInit();
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("schedule flow func init failed, ret=%d", ret);
            Stop();
            return;
        }
        return;
    }

    UDF_RUN_LOG_INFO("Start to schedule FlowFunc.");
    for (size_t idx = 0UL; idx < func_processors_.size(); ++idx) {
        int32_t ret = ScheduleFlowFunc(idx);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("ScheduleFlowFunc failed, ret=%d, idx=%zu, flowFuncInfo=%s.",
                ret,
                idx,
                func_processors_[idx]->GetFlowFuncInfo().c_str());
            Stop();
            return;
        }
    }
    ControlMessageType msg_type = ControlMessageType::kUnknow;
    UDF_RUN_LOG_INFO("request_queue_wrapper_ %d.", request_queue_wrapper_ == nullptr);
    if ((request_queue_wrapper_ != nullptr) && (ProcessRequestMessageQueue(msg_type) != FLOW_FUNC_SUCCESS)) {
        SendMessageByResponseQueue(msg_type, FLOW_FUNC_FAILED);
        UDF_LOG_ERROR("Try proccess request message queue data during init procedure failed.");
        Stop();
        return;
    }
    UDF_RUN_LOG_INFO("FlowFunc init event process end");
}

int32_t FlowFuncExecutor::SubscribeInputQueue() const {
    QueueSetInputPara input_param;
    QueueSetInput input;
    for (const auto &dev_input : dev_input_queue_map_) {
        uint32_t queue_device_id = dev_input.first;
        for (const auto &queue : dev_input.second) {
            uint32_t queue_id = queue.first;
            bool is_proxy_queue = queue.second;
            auto drv_ret = halQueueAttach(queue_device_id, queue_id, kAttachWaitTimeout);
            if (drv_ret != DRV_ERROR_NONE) {
                UDF_LOG_ERROR("attached input queue[%u] failed, ret[%d], queue_device_id[%u]", queue_id,
                    static_cast<int32_t>(drv_ret), queue_device_id);
                return FLOW_FUNC_ERR_QUEUE_ERROR;
            }

            // proxy queue no need sub queue event and work mode
            if (!is_proxy_queue) {
                input.queSetWorkMode.qid = queue_id;
                input.queSetWorkMode.workMode = static_cast<uint32_t>(QUEUE_MODE_PULL);
                input_param.inBuff = static_cast<void *>(&input);
                input_param.inLen = static_cast<uint32_t>(sizeof(QueueSetInput));
                (void)halQueueSet(queue_device_id, QUEUE_SET_WORK_MODE, &input_param);
                if (flow_msg_queues_.count(queue_id) > 0U) {
                    // for stream input, main thread no need subscribe queue
                    continue;
                }
                drv_ret = halQueueSubscribe(queue_device_id, queue_id, GlobalConfig::Instance().GetMainSchedGroupId(),
                    static_cast<int32_t>(QUEUE_TYPE_SINGLE));
                if (drv_ret != DRV_ERROR_NONE) {
                    UDF_LOG_ERROR("Failed to subscribe event for queue[%u], ret[%d].", queue_id,
                        static_cast<int32_t>(drv_ret));
                    return FLOW_FUNC_ERR_QUEUE_ERROR;
                }
                UDF_LOG_INFO("subscribe input queue end, queue_id=%u.", queue_id);
            }
        }
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::UnsubscribeInputQueue() const {
    for (const auto &dev_input : dev_input_queue_map_) {
        uint32_t queue_device_id = dev_input.first;
        for (const auto &queue : dev_input.second) {
            uint32_t queue_id = queue.first;
            if (flow_msg_queues_.count(queue_id) > 0U) {
                continue;
            }
            bool is_proxy_queue = queue.second;
            // proxy queue no need sub queue event and work mode
            if (!is_proxy_queue) {
                (void)halQueueUnsubscribe(queue_device_id, queue_id);
                UDF_LOG_INFO("unsubscribe input queue end, queue_id=%u.", queue_id);
            }
        }
    }
}

void FlowFuncExecutor::UnsubscribeOutputQueue() const {
    for (const auto &dev_output : dev_output_queue_map_) {
        uint32_t queue_device_id = dev_output.first;
        for (const auto &queue : dev_output.second) {
            uint32_t queue_id = queue.first;
            bool is_proxy_queue = queue.second;
            if (!is_proxy_queue) {
                (void)halQueueUnsubF2NFEvent(queue_device_id, queue_id);
                UDF_LOG_INFO("unsubscribe output queue end, queue_id=%u.", queue_id);
            }
        }
    }
}

int32_t FlowFuncExecutor::SubscribeOutputQueue() const {
    for (const auto &dev_output : dev_output_queue_map_) {
        uint32_t queue_device_id = dev_output.first;
        for (const auto &queue : dev_output.second) {
            uint32_t queue_id = queue.first;
            bool is_proxy_queue = queue.second;
            // time out
            auto drv_ret = halQueueAttach(queue_device_id, queue_id, kAttachWaitTimeout);
            if (drv_ret != DRV_ERROR_NONE) {
                UDF_LOG_ERROR("attached output queue[%u] failed, ret[%d], queue_device_id[%u]", queue_id,
                    static_cast<int32_t>(drv_ret), queue_device_id);
                return FLOW_FUNC_ERR_QUEUE_ERROR;
            }
            // proxy queue no need sub queue event
            if (!is_proxy_queue) {
                drv_ret = halQueueSubF2NFEvent(queue_device_id, queue_id, GlobalConfig::Instance().GetMainSchedGroupId());
                if (drv_ret != DRV_ERROR_NONE) {
                    UDF_LOG_ERROR("Failed to subscribe F2NF event for queue[%u], ret[%d].", queue_id,
                        static_cast<int32_t>(drv_ret));
                    return FLOW_FUNC_ERR_QUEUE_ERROR;
                }
                UDF_LOG_INFO("subscribe out queue F2NF end, queue_id=%u.", queue_id);
            }
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::SubscribeStatusOutputQueue() const {
    for (const auto &status_output_queue : status_output_queue_map_) {
        const int32_t queue_device_id = status_output_queue.first;
        for (const auto queue_id : status_output_queue.second) {
            const auto drv_ret = halQueueAttach(queue_device_id, queue_id, kAttachWaitTimeout);
            if (drv_ret != DRV_ERROR_NONE) {
                UDF_LOG_ERROR("attached status output queue[%u] failed, ret[%d], queue_device_id[%d]",
                    queue_id, static_cast<int32_t>(drv_ret), queue_device_id);
                return FLOW_FUNC_ERR_QUEUE_ERROR;
            }
            UDF_LOG_INFO("attached status output queue[%u] success.", queue_id);
        }
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::ProcessFlowFuncExecuteEvent(const struct event_info &event, uint32_t thread_idx) {
    auto flow_func_processor_idx = static_cast<size_t>(event.comm.subevent_id);

    UDF_LOG_DEBUG(
        "Flow func execute event begin, flow_func_processor_idx=%zu, thread_idx=%u.", flow_func_processor_idx, thread_idx);
    if (flow_func_processor_idx >= func_processors_.size()) {
        UDF_LOG_ERROR("FlowFuncExecute event invalid, flow_func_processor_idx=%zu.", flow_func_processor_idx);
        return;
    }
    // Get flow func processor
    auto flow_func_processor = func_processors_[flow_func_processor_idx];
    bool is_need_sched = flow_func_processor->Schedule(thread_idx);
    DoScheduleFlowFunc(is_need_sched, flow_func_processor_idx);
    UDF_LOG_DEBUG("Flow func execute event end.");
}

void FlowFuncExecutor::DoScheduleFlowFunc(bool is_need_sched, uint32_t flow_func_processor_idx) {
    if (is_need_sched) {
        (void)ScheduleFlowFunc(flow_func_processor_idx);
    } else {
        auto flow_func_processor = func_processors_[flow_func_processor_idx];
        if (!(flow_func_processor->IsOk())) {
            UDF_LOG_ERROR("Flow func executor will exit as flow_func_processor schedule failed, flowFuncInfo=%s.",
                flow_func_processor->GetFlowFuncInfo().c_str());
            Stop();
        }
    }
}

void FlowFuncExecutor::ProcessEmptyToNotEmptyEvent(const struct event_info &event, uint32_t thread_idx) {
    const uint32_t queue_id = event.comm.subevent_id;
    UDF_LOG_DEBUG("EmptyToNotEmptyEvent, queue_id=%u, thread_idx=%u.", queue_id, thread_idx);
    if (queue_id == GlobalConfig::Instance().GetReqQueueId()) {
        ControlMessageType msg_type = ControlMessageType::kUnknow;
        const auto ret = ProcessRequestMessageQueue(msg_type);
        if (ret != FLOW_FUNC_SUCCESS) {
            SendMessageByResponseQueue(msg_type, ret);
            Stop();
            UDF_LOG_ERROR("Process request queue = %u message failed. Start to exit.", queue_id);
        }
        return;
    }

    std::map<uint32_t, size_t>::const_iterator iter = input_to_flow_func_processor_idx_.find(queue_id);
    if (iter == input_to_flow_func_processor_idx_.cend()) {
        UDF_LOG_WARN("skip processing E2NE event for input queue[%u].", queue_id);
        return;
    }
    auto flow_func_idx = iter->second;
    if (flow_func_idx >= func_processors_.size()) {
        UDF_LOG_ERROR("flow_func_idx is invalid, queue_id=%u, flow_func_idx=%zu, funcProcessors size=%zu",
            queue_id,
            flow_func_idx,
            func_processors_.size());
        return;
    }
    auto flow_func_processor = func_processors_[flow_func_idx];
    bool is_need_sched = flow_func_processor->EmptyToNotEmpty();
    if (is_need_sched) {
        (void)ScheduleFlowFunc(flow_func_idx);
    }
}

int32_t FlowFuncExecutor::DequeueAndParseRequestMessage(ff::deployer::ExecutorRequest &req_msg,
    RequestMsgType &msg_type) const {
    Mbuf *control_muff = nullptr;

    auto wrapper_ret = request_queue_wrapper_->Dequeue(control_muff);
    if (wrapper_ret == HICAID_ERR_QUEUE_EMPTY) {
        UDF_LOG_INFO("Message queue turns to empty status.");
        return FLOW_FUNC_ERR_QUEUE_EMPTY;
    }
    if ((wrapper_ret != HICAID_SUCCESS)) {
        UDF_LOG_ERROR("Dequeue message in control queue failed, ret = %d.", wrapper_ret);
        return FLOW_FUNC_FAILED;
    }
    auto mbuf_deleter = [control_muff]() { (void)halMbufFree(control_muff); };
    ScopeGuard mbuf_guard(mbuf_deleter);
    return ParseRequestMessage(control_muff, req_msg, msg_type);
}

int32_t FlowFuncExecutor::ParseRequestMessage(Mbuf *control_mbuf, ff::deployer::ExecutorRequest &req_msg,
    RequestMsgType &msg_type) const {
    void *data_ptr = nullptr;
    auto drv_ret = halMbufGetBuffAddr(control_mbuf, &data_ptr);
    if ((drv_ret != DRV_ERROR_NONE) || (data_ptr == nullptr)) {
        UDF_LOG_ERROR("Failed to get data or data is nullptr, ret[%d].", drv_ret);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    uint64_t data_len = 0UL;
    drv_ret = halMbufGetDataLen(control_mbuf, &data_len);
    if ((drv_ret != DRV_ERROR_NONE) || (data_len == 0UL)) {
        UDF_LOG_ERROR("Failed to get data or data length is 0, ret[%d].", drv_ret);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    google::protobuf::io::ArrayInputStream stream(data_ptr, static_cast<int32_t>(data_len));
    const auto parse_ret = req_msg.ParseFromZeroCopyStream(&stream);
    if (!parse_ret) {
        UDF_LOG_ERROR("Parse control message failed. data_len=%lu.", data_len);
        return FLOW_FUNC_FAILED;
    }
    if (req_msg.has_clear_model_message()) {
        msg_type = RequestMsgType::kControlMsg;
        UDF_LOG_DEBUG("Current message is control message.");
    } else if (req_msg.has_exception_request()) {
        msg_type = RequestMsgType::kExceptionMsg;
        UDF_LOG_DEBUG("Current message is exception message.");
    } else if (req_msg.type() == ff::deployer::ExecutorRequestType::kNotify) {
        msg_type = RequestMsgType::kNotify;
        UDF_LOG_DEBUG("Current message is notify message.");
    } else {
        UDF_LOG_ERROR("Request msg queue should only send control message, exception info and notify message.");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::ProcessControlMsg(ff::deployer::ExecutorRequest_ClearModelRequest &ctrl_msg) {
    if (ctrl_msg.clear_msg_type() == static_cast<int32_t>(ControlMessageType::kSuspend)) {
        GlobalConfig::Instance().SetAbnormalStatus(true);
        UDF_LOG_INFO("Executor will send suspend for processor num[%zu].", func_processors_.size());
        for (size_t i = 0UL; i < func_processors_.size(); ++i) {
            func_processors_[i]->SetClearAndSuspend();
            std::unique_lock<std::mutex> lk(suspend_mutex_);
            (void)suspend_process_ids_.insert(static_cast<uint32_t>(i));
            auto ret = ScheduleFlowFunc(i);
            if (ret != FLOW_FUNC_SUCCESS) {
                return ret;
            }
        }
    } else if (ctrl_msg.clear_msg_type() == static_cast<int32_t>(ControlMessageType::kRecover)) {
        GlobalConfig::Instance().SetAbnormalStatus(true);
        UDF_LOG_INFO("Executor will send recover for processor num[%zu].", func_processors_.size());
        for (size_t i = 0UL; i < func_processors_.size(); ++i) {
            func_processors_[i]->SetClearAndRecover();
            std::unique_lock<std::mutex> lk(recover_mutex_);
            (void)recover_process_ids_.insert(static_cast<uint32_t>(i));
            auto ret = ScheduleFlowFunc(i);
            if (ret != FLOW_FUNC_SUCCESS) {
                return ret;
            }
        }
    } else {
        UDF_LOG_ERROR("Invalid message type got from control queue. msg type :%d", ctrl_msg.clear_msg_type());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::ProcessExceptionMsg(const ff::deployer::ExecutorRequest_DataflowExceptionNotify &exp_msg) {
    UDF_LOG_INFO("Executor will send exception info for processor num[%zu].", func_processors_.size());
    if (exp_msg.type() == static_cast<uint32_t>(ExceptionType::kAddException)) {
        UdfExceptionInfo exp_info = {};
        exp_info.exp_code = exp_msg.exception_code();
        exp_info.trans_id = exp_msg.trans_id();
        exp_info.user_context_id = exp_msg.user_context_id();
        auto ret = memcpy_s(exp_info.exp_context, kMaxMbufHeadLen,
                            exp_msg.exception_context().data(), exp_msg.exception_context().size());
        if (ret != EOK) {
            UDF_LOG_ERROR("Copy exception info from message failed. ret=%d, trans_id[%lu], exp_code[%d],"
                          "user_context_id[%lu]", static_cast<int32_t>(ret), exp_info.trans_id, exp_info.exp_code,
                          exp_info.user_context_id);
            return FLOW_FUNC_FAILED;
        }
        exp_info.exp_context_size = exp_msg.exception_context().size();
        for (size_t i = 0UL; i < func_params_.size(); ++i) {
            if (func_params_[i]->HandleInvokedException(exp_msg.scope(), exp_msg.trans_id(), true)) {
                continue;
            }
        }
        for (size_t i = 0UL; i < func_processors_.size(); ++i) {
            if (!(func_processors_[i]->CheckSameScope(exp_msg.scope()))) {
                return FLOW_FUNC_SUCCESS;
            }
            func_processors_[i]->RecordExceptionInfo(exp_info);
            ret = ScheduleFlowFunc(i);
            if (ret != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Schedule func failed for trans_id[%lu], exp_code[%d], user_context_id[%lu]",
                              exp_info.trans_id, exp_info.exp_code, exp_info.user_context_id);
                return ret;
            }
        }
    } else if (exp_msg.type() == static_cast<uint32_t>(ExceptionType::kDeleteException)) {
        for (size_t i = 0UL; i < func_params_.size(); ++i) {
            if (func_params_[i]->HandleInvokedException(exp_msg.scope(), exp_msg.trans_id(), false)) {
                continue;
            }
        }
        for (size_t i = 0UL; i < func_processors_.size(); ++i) {
            if (!(func_processors_[i]->CheckSameScope(exp_msg.scope()))) {
                return FLOW_FUNC_SUCCESS;
            }
            func_processors_[i]->RecordDeleteException(exp_msg.trans_id());
        }
    } else {
        UDF_LOG_ERROR("Invalid Type[%u] Get from exception request.", exp_msg.type());
        return FLOW_FUNC_FAILED;
    }

    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::ProcessRequestMessageQueue(ControlMessageType &ctrl_msg_type) {
    int32_t deq_ret = FLOW_FUNC_ERR_QUEUE_EMPTY;
    bool ctrl_msg_proc = false;
    do {
        ff::deployer::ExecutorRequest req_msg;
        RequestMsgType msg_type;
        deq_ret = DequeueAndParseRequestMessage(req_msg, msg_type);
        if (deq_ret == FLOW_FUNC_ERR_QUEUE_EMPTY) {
            UDF_LOG_DEBUG("Now message queue is empty.");
            break;
        }
        if (deq_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Parse request message failed.");
            return deq_ret;
        }
        if (msg_type == RequestMsgType::kExceptionMsg) {
            ctrl_msg_type = ControlMessageType::kException;
            auto exp_msg = req_msg.exception_request().exception_notify();
            if (ProcessExceptionMsg(exp_msg) != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Process exception msg failed.");
                return FLOW_FUNC_FAILED;
            }
            if (SendMessageByResponseQueue(ControlMessageType::kException, FLOW_FUNC_SUCCESS) != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Send response message in queue failed.");
                return FLOW_FUNC_FAILED;
            }
        } else {
            // The clear and suspend messages are processed only once each time when the message queue is not empty.
            if (ctrl_msg_proc) {
                continue;
            }
            ctrl_msg_proc = true;
            ff::deployer::ExecutorRequest_ClearModelRequest ctrl_msg = req_msg.clear_model_message();
            ctrl_msg_type = static_cast<ControlMessageType>(ctrl_msg.clear_msg_type());
            if (ProcessControlMsg(ctrl_msg) != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Process control msg failed.");
                return FLOW_FUNC_FAILED;
            }
        }
    } while (deq_ret != FLOW_FUNC_ERR_QUEUE_EMPTY);
    return FLOW_FUNC_SUCCESS;
}

template<typename T>
int32_t FlowFuncExecutor::SerializeProtoToMbuf(const T &proto_msg, Mbuf *&mbuf_to_generate) {
    const size_t rsp_size = proto_msg.ByteSizeLong();
    const FillFunc fill_func = [&proto_msg](void *const buffer, const size_t size) {
        if (proto_msg.SerializeToArray(buffer, static_cast<int32_t>(size))) {
            return FLOW_FUNC_SUCCESS;
        }
        UDF_LOG_ERROR("Protobuf serializeToArray failed.");
        return FLOW_FUNC_FAILED;
    };
    Mbuf *mbuf = nullptr;
    int32_t ret = GenerateMbuf(rsp_size, fill_func, mbuf);
    if ((ret != FLOW_FUNC_SUCCESS) || (mbuf == nullptr)) {
        UDF_LOG_ERROR("generate mbuf failed, ret[%d].", ret);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    mbuf_to_generate = mbuf;
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncExecutor::SendMessageByResponseQueue(const ControlMessageType &msg_type, const int32_t result) {
    if (GlobalConfig::Instance().GetRspQueueId() == UINT32_MAX) {
        UDF_LOG_INFO("There is not message queue in current version. skip to send message.");
        return FLOW_FUNC_SUCCESS;
    }
    std::string msg;
    const auto iter = kResponseMsgMap.find(msg_type);
    if (iter != kResponseMsgMap.cend()) {
        msg = iter->second;
    } else {
        msg = "Unknown operator " + std::to_string(static_cast<int32_t>(msg_type));
    }
    ff::deployer::ExecutorResponse response;
    response.set_error_code(result);
    if (result == FLOW_FUNC_SUCCESS) {
        msg += " success.";
    } else {
        msg += " failed.";
    }
    response.set_error_message(msg);
    Mbuf *mbuf = nullptr;
    if (SerializeProtoToMbuf<ff::deployer::ExecutorResponse>(response, mbuf) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Serialize proto to mbuf failed.");
        return FLOW_FUNC_FAILED;
    }
    UDF_LOG_INFO("Prepare to send response message.code[%d], msg[%s].", result, msg.c_str());
    const auto ret = response_queue_wrapper_->Enqueue(mbuf);
    if (ret != HICAID_SUCCESS) {
        UDF_LOG_ERROR("Enqueue message buffer to response queue failed. queue_id = %u, ret = %d.",
                      GlobalConfig::Instance().GetRspQueueId(), ret);
        (void)halMbufFree(mbuf);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::ProcessFullToNoFullEvent(const struct event_info &event, uint32_t thread_idx) {
    const uint32_t queue_id = event.comm.subevent_id;
    UDF_LOG_DEBUG("FullToNoFullEvent, queue_id=%u, thread_idx=%u.", queue_id, thread_idx);
    std::map<uint32_t, std::vector<size_t>>::const_iterator context_iter = output_to_flow_func_processor_idx_.find(queue_id);
    if (context_iter == output_to_flow_func_processor_idx_.cend()) {
        UDF_LOG_WARN("there is no flow func use output queue_id=%u", queue_id);
        return;
    }
    // one output queue may have multi func
    for (auto flow_func_idx : context_iter->second) {
        if (flow_func_idx >= func_processors_.size()) {
            UDF_LOG_ERROR("flow_func_idx is invalid, queue_id=%u, flow_func_idx=%zu, flow_func_processor size=%zu",
                queue_id,
                flow_func_idx,
                func_processors_.size());
            return;
        }
        auto flow_func_processor = func_processors_[flow_func_idx];
        bool is_need_sched = flow_func_processor->FullToNotFull();
        if (is_need_sched) {
            (void)ScheduleFlowFunc(flow_func_idx);
        }
    }
}

void FlowFuncExecutor::ProcessTimerEvent(const struct event_info &event, uint32_t thread_idx) {
    uint32_t timer_id = event.comm.subevent_id;
    UDF_LOG_DEBUG("ProcessTimerEvent enter, timer_id=%u, thread_idx=%u.", timer_id, thread_idx);
    FlowFuncTimer::Instance().ExecCallBack(timer_id);
}

void FlowFuncExecutor::ProcessNotifyThreadExitEvent(const struct event_info &event, uint32_t thread_idx) {
    // do nothing, as notify has set run flag to false. event just used for wake up event wait.
    (void)event;
    UDF_LOG_INFO("Thread[%u] receive notify thread exit event.", thread_idx);
}

void FlowFuncExecutor::ConstructException(const std::string &current_scope,
                                          const UdfExceptionInfo &exception_info,
                                          ff::deployer::SubmodelStatus &exception_msg) {
    exception_msg.set_msg_type(static_cast<uint32_t>(StatusQueueMsgType::kRaiseExceptionMsgType));
    auto exception_proto = exception_msg.mutable_exception_info();
    exception_proto->set_trans_id(exception_info.trans_id);
    exception_proto->set_exception_code(exception_info.exp_code);
    exception_proto->set_scope(current_scope);
    exception_proto->set_user_context_id(exception_info.user_context_id);
    exception_proto->set_exception_context(&exception_info.exp_context[0], exception_info.exp_context_size);
    UDF_LOG_INFO("Construct exception message: trans_id[%lu] exp_code[%d] scope[%s] success.",
                 exception_info.trans_id, exception_info.exp_code, current_scope.c_str());
}

void FlowFuncExecutor::ProcessRaiseExceptionEvent(const struct event_info &event, uint32_t thread_idx) {
    auto flow_func_processor_idx = static_cast<size_t>(event.comm.subevent_id);
    UDF_LOG_DEBUG("Raise exception event begin, flow_func_processor_idx=%zu, thread_idx=%u.",
        flow_func_processor_idx, thread_idx);
    if (flow_func_processor_idx >= func_processors_.size()) {
        UDF_LOG_ERROR("Raise exception event invalid, flow_func_processor_idx=%zu.", flow_func_processor_idx);
        Stop();
        return;
    }
    if (event.priv.msg_len != sizeof(uint64_t)) {
        UDF_LOG_ERROR("Get processor event msg length invalid, flow_func_processor_idx=%zu.", flow_func_processor_idx);
        Stop();
        return;
    }
    const uint64_t trans_id = *(reinterpret_cast<const uint64_t *>(event.priv.msg));
    ReportExceptionMbufGenFunc mbuf_gen_func = [this](const std::string &current_scope,
                                                    const UdfExceptionInfo &exception_info,
                                                    Mbuf *&mbuf_to_generate) -> int32_t {
        ff::deployer::SubmodelStatus exception_msg;
        ConstructException(current_scope, exception_info, exception_msg);
        Mbuf *mbuf = nullptr;
        if (SerializeProtoToMbuf<ff::deployer::SubmodelStatus>(exception_msg, mbuf) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Serialize proto to mbuf failed.");
            return FLOW_FUNC_FAILED;
        }
        mbuf_to_generate = mbuf;
        return FLOW_FUNC_SUCCESS;
    };
    if (func_processors_[flow_func_processor_idx]->WriteStatusOutputQueue(trans_id, mbuf_gen_func) != FLOW_FUNC_SUCCESS) {
        UDF_RUN_LOG_ERROR("Processor[%zu] report exception msg failed. Start to stop running.", flow_func_processor_idx);
        Stop();
        return;
    }
}

void FlowFuncExecutor::ProcessReportStatusEvent(const struct event_info &event, uint32_t thread_idx) {
    auto flow_func_processor_idx = static_cast<size_t>(event.comm.subevent_id);
    UDF_LOG_DEBUG("Report status event begin, flow_func_processor_idx=%zu, thread_idx=%u.",
        flow_func_processor_idx, thread_idx);
    if (flow_func_processor_idx >= func_processors_.size()) {
        UDF_LOG_ERROR("Report status event invalid, flow_func_processor_idx=%zu.", flow_func_processor_idx);
        Stop();
        return;
    }
    const auto ret = ReportStatus(flow_func_processor_idx);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Report status failed, flow_func_processor_idx=%zu, thread_idx=%u.",
            flow_func_processor_idx, thread_idx);
        Stop();
        return;
    }
    UDF_LOG_DEBUG("Report status event end, flow_func_processor_idx=%zu, thread_idx=%u.",
        flow_func_processor_idx, thread_idx);
}

int32_t FlowFuncExecutor::ReportStatus(const size_t flow_func_processor_idx) {
    ReportStatusMbufGenFunc mbuf_gen_func = [this](const std::vector<QueueDevInfo> &input_queue_infos,
                                                 const uint32_t model_uuid,
                                                 const uint32_t input_consume_sum,
                                                 Mbuf *&mbuf_to_generate) -> int32_t {
        // construct SubmodelStatus protobuf object
        ff::deployer::SubmodelStatus sub_model_status;
        sub_model_status.set_model_uuid(model_uuid);
        for (const auto input_queue_info : input_queue_infos) {
            const auto input_queue_id = input_queue_info.queue_id;
            uint32_t depth_value = UINT32_MAX;
            QueueInfo info;
            const auto drv_ret = halQueueQueryInfo(input_queue_info.device_id, input_queue_id, &info);
            if (drv_ret != DRV_ERROR_NONE) {
                UDF_LOG_WARN("query queue info failed, queue id[%u], device id[%u], ret[%d].",
                    input_queue_id, input_queue_info.device_id, static_cast<int32_t>(drv_ret));
            } else {
                depth_value = static_cast<size_t>(info.size);
            }
            auto queue_status = sub_model_status.add_queue_statuses();
            queue_status->set_queue_depth(depth_value);
            queue_status->set_input_consume_num(input_consume_sum);
            auto queue_attrs = queue_status->mutable_queue_attrs();
            queue_attrs->set_queue_id(input_queue_id);
            queue_attrs->set_device_type(input_queue_info.device_type);
            queue_attrs->set_device_id(input_queue_info.device_id);
            queue_attrs->set_logic_id(input_queue_info.logic_queue_id);
        }
        sub_model_status.set_msg_type(static_cast<uint32_t>(StatusQueueMsgType::kReportStatusMsgType));
        // generate mbuf
        if (SerializeProtoToMbuf<ff::deployer::SubmodelStatus>(sub_model_status, mbuf_to_generate) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Serialize proto to mbuf failed.");
            return FLOW_FUNC_FAILED;
        }
        UDF_LOG_INFO("Generate report status mbuf success, status[%s].",
            sub_model_status.DebugString().c_str());
        return FLOW_FUNC_SUCCESS;
    };
    return func_processors_[flow_func_processor_idx]->WriteStatusOutputQueue(mbuf_gen_func);
}

int32_t FlowFuncExecutor::CheckProcessorEventParams(const struct event_info &event) {
    const auto flow_func_processor_idx = static_cast<size_t>(event.comm.subevent_id);
    if (flow_func_processor_idx >= func_processors_.size()) {
        UDF_LOG_ERROR("Report processor id invalid, flow_func_processor_idx=%zu.", flow_func_processor_idx);
        return FLOW_FUNC_PROCESSOR_PARAM_ERROR;
    }
    if (event.priv.msg_len != sizeof(int32_t)) {
        UDF_LOG_ERROR("Get processor event msg length invalid, flow_func_processor_idx=%zu.", flow_func_processor_idx);
        Stop();
        return FLOW_FUNC_PROCESSOR_PARAM_ERROR;
    }
    const int32_t ret = *(reinterpret_cast<const int32_t *>(event.priv.msg));
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Processor event result is %d, flow_func_processor_idx=%zu.", ret, flow_func_processor_idx);
        return ret;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::ProcessReportSuspendEvent(const struct event_info &event, uint32_t thread_idx) {
    const auto flow_func_processor_idx = static_cast<size_t>(event.comm.subevent_id);
    auto ret = CheckProcessorEventParams(event);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Get processor:%zu suspend finished event is invalid. result:%d", flow_func_processor_idx, ret);
        SendMessageByResponseQueue(ControlMessageType::kSuspend, ret);
        Stop();
        return;
    }
    std::unique_lock<std::mutex> lk(suspend_mutex_);
    (void)suspend_process_ids_.erase(flow_func_processor_idx);
    if (suspend_process_ids_.empty()) {
        ret = FlowFuncManager::Instance().ResetFuncState();
        if (ret != FLOW_FUNC_SUCCESS) {
            // if not support reset state, need recreate func when recover.
            FlowFuncManager::Instance().Reset();
            for (auto &funcProcessor : func_processors_) {
                funcProcessor->ReleaseFuncWrapper();
            }
        }
        if (SendMessageByResponseQueue(ControlMessageType::kSuspend, FLOW_FUNC_SUCCESS) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Send response message in queue failed.");
            Stop();
        }
    }
    UDF_LOG_INFO("Process suspend finish event, flow_func_processor_idx=%zu, thread_idx=%u.",
        flow_func_processor_idx, thread_idx);
}

void FlowFuncExecutor::ProcessReportRecoverEvent(const struct event_info &event, uint32_t thread_idx) {
    const auto flow_func_processor_idx = static_cast<size_t>(event.comm.subevent_id);
    const auto ret = CheckProcessorEventParams(event);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Get processor:%zu recover finished event is invalid. result:%d", flow_func_processor_idx, ret);
        SendMessageByResponseQueue(ControlMessageType::kRecover, ret);
        Stop();
        return;
    }
    std::unique_lock<std::mutex> lk(recover_mutex_);
    (void)recover_process_ids_.erase(flow_func_processor_idx);
    if (recover_process_ids_.empty()) {
        if (SendMessageByResponseQueue(ControlMessageType::kRecover, FLOW_FUNC_SUCCESS) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Send response message in queue failed.");
            Stop();
        }
    }
    GlobalConfig::Instance().SetAbnormalStatus(false);
    UDF_LOG_INFO("Process recover finish event end, flow_func_processor_idx=%zu, thread_idx=%u.",
        flow_func_processor_idx, thread_idx);
}

void FlowFuncExecutor::ProcessSwitchSoftModeEvent(const struct event_info &event, uint32_t thread_idx) {
    (void)event;
    (void)thread_idx;
    UDF_LOG_INFO("Udf event switch soft sched mode.");
}

int32_t FlowFuncExecutor::GenerateMbuf(const size_t req_size, const FillFunc &fill_func, Mbuf *&mbuf_to_generate) {
    // alloc mbuf
    Mbuf *mbuf = nullptr;
    auto drv_ret = halMbufAlloc(req_size, &mbuf);
    if ((drv_ret != DRV_ERROR_NONE) || (mbuf == nullptr)) {
        UDF_LOG_ERROR("Alloc mbuff failed, drv_ret=%d, dataSize=%lu.", drv_ret, req_size);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    auto mbuf_deleter = [mbuf]() { (void)halMbufFree(mbuf); };
    ScopeGuard mbuf_guard(mbuf_deleter);
    drv_ret = halMbufSetDataLen(mbuf, req_size);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Mbuff set data length failed, drv_ret=%d, dataSize=%lu.", drv_ret, req_size);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    // write mbuf data
    void *buf_addr = nullptr;
    drv_ret = halMbufGetBuffAddr(mbuf, &buf_addr);
    if (drv_ret != DRV_ERROR_NONE || buf_addr == nullptr) {
        UDF_LOG_ERROR("Failed to get buff addr, ret[%d].", drv_ret);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    const auto fill_ret = fill_func(buf_addr, req_size);
    if (fill_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to fill mbuf data, ret[%d].", fill_ret);
        return fill_ret;
    }
    mbuf_guard.ReleaseGuard();
    mbuf_to_generate = mbuf;
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncExecutor::DumpMetrics(bool with_queue_info) const {
    for (const auto &funcProcessor : func_processors_) {
        funcProcessor->DumpFlowFuncInfo(with_queue_info);
        funcProcessor->DumpModelMetrics(with_queue_info);
    }
}
}
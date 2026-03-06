/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/helper/om_file_helper.h"

#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/helper/model_parser_base.h"
#include "common/helper/file_saver.h"
#include "common/math/math_util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "ge/ge_error_codes.h"
#include "runtime/rt.h"

namespace {
constexpr uint32_t kOptionalNum = 2U;
constexpr uint32_t kMaxIncreasePartitionNum = 2U;  // 新增partition类型时需要修改
constexpr const char_t *kSocVersion = "soc_version";
constexpr const char_t *kArchType = "arch_type";
}
namespace ge {
static bool IsPartitionTableNumValid(const uint32_t partition_num, const uint32_t increase_partition_num) {
  if ((partition_num != (PARTITION_SIZE + increase_partition_num)) &&
      (partition_num != (PARTITION_SIZE - 1U + increase_partition_num)) &&
      (partition_num != (PARTITION_SIZE - kOptionalNum + increase_partition_num)) &&
      (partition_num != (1U + increase_partition_num))) {
    GELOGW("Invalid partition_table->num:%u", partition_num);
    return false;
  }
  return true;
}

bool OmFileLoadHelper::CheckPartitionTableNum(const uint32_t partition_num) {
  bool is_partition_num_valid = false;
  for (uint32_t i = 0U; i <= kMaxIncreasePartitionNum; ++i) {
    if (IsPartitionTableNumValid(partition_num, i)) {
      is_partition_num_valid = true;
      break;
    }
  }
  return is_partition_num_valid;
}

// For Load
Status OmFileLoadHelper::Init(const ModelData &model) {
  uint64_t model_len = 0UL;
  uint8_t *model_data = nullptr;
  GE_CHK_STATUS_RET_NOLOG(ModelParserBase::ParseModelContent(model, model_data, model_len));
  ModelFileHeader *file_header = PtrToPtr<void, ModelFileHeader>(model.model_data);
  return Init(model_data, model_len, file_header);
}

Status OmFileLoadHelper::Init(uint8_t *const model_data, const uint32_t model_data_size) {
  return Init(model_data, static_cast<uint64_t>(model_data_size), nullptr);
}

Status OmFileLoadHelper::Init(uint8_t *const model_data, const uint32_t model_data_size, const uint32_t model_num) {
  return Init(model_data, static_cast<uint64_t>(model_data_size), model_num, nullptr);
}

Status OmFileLoadHelper::Init(uint8_t *const model_data,
                              const uint64_t model_data_size,
                              const ModelFileHeader *file_header) {
  size_t mem_offset = 0U;
  const Status status = LoadModelPartitionTable(model_data, model_data_size, 0U, mem_offset, file_header);
  if (status != SUCCESS) {
    return status;
  }
  is_inited_ = true;
  return SUCCESS;
}

Status OmFileLoadHelper::Init(uint8_t *const model_data,
                              const uint64_t model_data_size,
                              const uint32_t model_num,
                              const ModelFileHeader *file_header) {
  const Status status = LoadModelPartitionTable(model_data, model_data_size, model_num, file_header);
  if (status != SUCCESS) {
    return status;
  }
  is_inited_ = true;
  return SUCCESS;
}

// Use both
Status OmFileLoadHelper::GetModelPartition(const ModelPartitionType type, ModelPartition &partition) {
  return GetModelPartition(type, partition, 0U);
}

Status OmFileLoadHelper::GetModelPartition(const ModelPartitionType type,
                                           ModelPartition &partition, const size_t model_index) const {
  if (!is_inited_) {
    GELOGE(PARAM_INVALID, "OmFileLoadHelper has not been initialized!");
    return PARAM_INVALID;
  }
  if (model_index >= model_contexts_.size()) {
    GELOGE(PARAM_INVALID, "cur index : %zu, model_contexts size:%zu", model_index, model_contexts_.size());
    return PARAM_INVALID;
  }
  const auto &cur_ctx = model_contexts_[model_index];
  for (const ModelPartition &part : cur_ctx.partition_datas_) {
    if (part.type == type) {
      partition = part;
      return SUCCESS;
    }
  }

  static const std::set<ModelPartitionType> model_partitions = {
      ModelPartitionType::TBE_KERNELS, ModelPartitionType::WEIGHTS_DATA,     ModelPartitionType::CUST_AICPU_KERNELS,
      ModelPartitionType::SO_BINS,     ModelPartitionType::MODEL_INOUT_INFO, ModelPartitionType::TASK_INFO,
      ModelPartitionType::TILING_DATA,
  };

  if (model_partitions.count(type) == 0UL) {
    GELOGE(FAILED, "GetModelPartition:type:%d is not in partition_datas!", static_cast<int32_t>(type));
    return FAILED;
  }
  return SUCCESS;
}

const std::vector<ModelPartition> &OmFileLoadHelper::GetModelPartitions(const size_t model_index) const {
  if (model_index >= model_contexts_.size()) {
    GELOGE(PARAM_INVALID, "cur index : %zu, model_contexts size:%zu", model_index, model_contexts_.size());
    static const std::vector<ModelPartition> kEmptyVec;
    return kEmptyVec;
  }
  return model_contexts_[model_index].partition_datas_;
}

static Status ConvertToModelPartitionTable(const TinyModelPartitionTable * const tiny_table,
                                           std::unique_ptr<uint8_t[]> &model_partition_table_holder) {
  const size_t total_size = sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * tiny_table->num;
  model_partition_table_holder = MakeUnique<uint8_t[]>(total_size);
  if (model_partition_table_holder == nullptr) {
    GELOGE(FAILED, "malloc failed for size %zu", total_size);
    return FAILED;
  }
  auto table = reinterpret_cast<ModelPartitionTable *>(model_partition_table_holder.get());
  table->num = tiny_table->num;
  for (size_t i = 0U; i < table->num; ++i) {
    table->partition[i].type = tiny_table->partition[i].type;
    table->partition[i].mem_offset = static_cast<uint64_t>(tiny_table->partition[i].mem_offset);
    table->partition[i].mem_size = static_cast<uint64_t>(tiny_table->partition[i].mem_size);
  }
  return SUCCESS;
}

Status OmFileLoadHelper::LoadModelPartitionTable(const uint8_t *const model_data,
                                                 const uint64_t model_data_size,
                                                 const size_t model_index,
                                                 size_t &mem_offset,
                                                 const ModelFileHeader *file_header) {
  if (model_data == nullptr) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID, "Param model_data must not be null!");
    return ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID;
  }

  if ((model_data_size < mem_offset) || (model_data_size - mem_offset <= sizeof(ModelPartitionTable))) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "The partition table size %zu is greater than model data size %lu",
           mem_offset + sizeof(ModelPartitionTable), model_data_size);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  const bool is_flow_model = (file_header != nullptr) && (file_header->modeltype == MODEL_TYPE_FLOW_MODEL);
  // Init partition table
  ModelPartitionTable *partition_table = nullptr;
  std::unique_ptr<uint8_t[]> model_partition_table_holder = nullptr;
  size_t partition_table_size = 0U;
  if (is_flow_model || ((file_header != nullptr) && (file_header->model_length != 0UL))) {
    partition_table = PtrToPtr<void, ModelPartitionTable>(ValueToPtr(PtrToValue(model_data) + mem_offset));
    partition_table_size = SizeOfModelPartitionTable(*partition_table);
  } else {
    TinyModelPartitionTable * const tiny_partition_table =
        PtrToPtr<void, TinyModelPartitionTable>(ValueToPtr(PtrToValue(model_data) + mem_offset));
    if (!CheckPartitionTableNum(tiny_partition_table->num)) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid tiny_partition_table->num:%u", tiny_partition_table->num);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    partition_table_size = SizeOfTinyModelPartitionTable(*tiny_partition_table);
    GE_CHK_STATUS_RET_NOLOG(ConvertToModelPartitionTable(tiny_partition_table, model_partition_table_holder));
    partition_table = reinterpret_cast<ModelPartitionTable *>(model_partition_table_holder.get());
  }

  if (is_flow_model) {
    constexpr uint32_t kMaxFlowModelPartitionNum = 4096U;
    if (partition_table->num > kMaxFlowModelPartitionNum) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid flow model partition_table->num:%u, range[0, %u]",
             partition_table->num, kMaxFlowModelPartitionNum);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  } else {
    // Davinici model partition include graph-info  weight-info  task-info  tbe-kernel :
    // Original model partition include graph-info
    if (!CheckPartitionTableNum(partition_table->num)) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid partition_table->num:%u", partition_table->num);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  }
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(mem_offset, partition_table_size));
  mem_offset += partition_table_size;
  GELOGD("Cur model index:%zu, ModelPartitionTable num:%u, ModelFileHeader size:%zu, ModelPartitionTable size:%zu",
         model_index, partition_table->num, sizeof(ModelFileHeader), partition_table_size);
  if (model_data_size <= mem_offset) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID, "invalid model data, partition_table->num:%u, data size %lu",
           partition_table->num, model_data_size);
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }

  if (model_index != model_contexts_.size()) {
    GELOGE(FAILED, "cur index is %zu make model_contexts_ overflow", model_index);
    return FAILED;
  }
  model_contexts_.push_back(OmFileContext{});
  for (uint32_t i = 0U; i < partition_table->num; i++) {
    ModelPartition partition;
    partition.size = partition_table->partition[i].mem_size;
    partition.data = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(model_data) + mem_offset));
    partition.type = partition_table->partition[i].type;
    model_contexts_[model_index].partition_datas_.push_back(partition);
    if ((partition.size > model_data_size) || (mem_offset > static_cast<size_t>(model_data_size - partition.size))) {
      GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
             "The partition size (%lu + %zu) is greater than the model data size %lu.",
             partition.size, mem_offset, model_data_size);
      return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
    }
    if (CheckUint64AddOverflow(mem_offset, partition.size) != SUCCESS) {
      GELOGE(FAILED, "UINT64 %zu and %lu addition can result in overflow!", mem_offset, partition.size);
      return FAILED;
    }
    mem_offset += partition.size;
    GELOGD("type:%d, size:%lu, index:%zu", static_cast<int32_t>(partition.type), partition.size, model_index);
  }
  return SUCCESS;
}

Status OmFileLoadHelper::LoadModelPartitionTable(const uint8_t *const model_data, const uint64_t model_data_size,
                                                 const uint32_t model_num, const ModelFileHeader *file_header) {
  if (model_data == nullptr) {
    GELOGE(PARAM_INVALID, "Param model_data must not be null!");
    return PARAM_INVALID;
  }

  size_t cur_offset = 0U;
  for (size_t index = 0U; index < static_cast<size_t>(model_num); ++index) {
    GE_CHK_STATUS_RET_NOLOG(LoadModelPartitionTable(model_data, model_data_size, index, cur_offset, file_header));
  }
  if (cur_offset != model_data_size) {
    GELOGE(FAILED, "do not get the complete model, read end offset:%zu, all size:%lu", cur_offset, model_data_size);
    return FAILED;
  }
  return SUCCESS;
}

Status OmFileLoadHelper::CheckModelCompatibility(const Model &model) const {
  std::string model_soc_version;
  std::string model_arch_type;
  (void) AttrUtils::GetStr(model, kSocVersion, model_soc_version);
  (void) AttrUtils::GetStr(model, kArchType, model_arch_type);
  GE_CHK_RT_RET(rtModelCheckCompatibility(model_soc_version.c_str(), model_arch_type.c_str()));
  GELOGD("soc_version:%s, arch_type:%s check valid", model_soc_version.c_str(), model_arch_type.c_str());
  return SUCCESS;
}

ModelPartitionTable *OmFileSaveHelper::GetPartitionTable() {
  return GetPartitionTable(0U);
}

ModelPartitionTable *OmFileSaveHelper::GetPartitionTable(const size_t cur_ctx_index,
                                                         const bool is_partition_align, const uint32_t align_bytes) {
  auto &cur_ctx = model_contexts_[cur_ctx_index];
  const uint64_t partition_size = static_cast<uint64_t>(cur_ctx.partition_datas_.size());
  // Build ModelPartitionTable, flex array
  cur_ctx.partition_table_.resize(sizeof(ModelPartitionTable) + (sizeof(ModelPartitionMemInfo) * partition_size),
                                  static_cast<char_t>(0));

  auto const partition_table = PtrToPtr<char_t, ModelPartitionTable>(cur_ctx.partition_table_.data());
  partition_table->num = static_cast<uint32_t>(partition_size);

  uint64_t mem_offset = 0UL;
  if (is_partition_align) {
    const auto table_size = (SizeOfModelPartitionTable(*partition_table));
    mem_offset = MemSizeAlign(table_size, align_bytes) - table_size;
    GELOGI("cur_ctx_index:%u, raw table size:%u, partition start offset:%u, align bytes:%u",
           cur_ctx_index, table_size, mem_offset, align_bytes);
  }

  for (size_t i = 0U; i < static_cast<size_t>(partition_size); i++) {
    const ModelPartition partition = cur_ctx.partition_datas_[i];
    partition_table->partition[i] = {partition.type, mem_offset, partition.size};
    if (CheckUint64AddOverflow(mem_offset, partition.size) != SUCCESS) {
      GELOGE(FAILED, "UINT64 %lu and %lu addition can result in overflow!", mem_offset, partition.size);
      return nullptr;
    }

    std::string bool_val = is_partition_align ? "true" : "false";
    GELOGD("Partition index:%u, type:%d, size:%lu, offset:%lu, is align:%s, align bytes:%u",
           i, static_cast<int32_t>(partition.type), partition.size, mem_offset,
           bool_val.c_str(), align_bytes);

    if (is_partition_align) {
      mem_offset += MemSizeAlign(partition.size, align_bytes);
    } else {
      mem_offset += partition.size;
    }
  }
  return partition_table;
}

Status OmFileSaveHelper::AddPartition(const ModelPartition &partition) {
  return AddPartition(partition, 0U);
}

Status OmFileSaveHelper::AddPartition(const ModelPartition &partition, const size_t cur_index) {
  if (cur_index >= model_contexts_.size()) {
    if (cur_index != model_contexts_.size()) {
      GELOGE(FAILED, "cur index is %zu make model_contexts_ overflow", cur_index);
      return FAILED;
    }
    OmFileContext tmp_ctx;
    tmp_ctx.model_data_len_ += partition.size;
    tmp_ctx.partition_datas_.push_back(partition);
    model_contexts_.push_back(tmp_ctx);
  } else {
    auto &cur_ctx = model_contexts_[cur_index];
    if (CheckUint64AddOverflow(cur_ctx.model_data_len_, partition.size) != SUCCESS) {
      GELOGE(FAILED, "UINT64 %lu and %lu addition can result in overflow!", cur_ctx.model_data_len_, partition.size);
      return FAILED;
    }
    cur_ctx.model_data_len_ += partition.size;
    cur_ctx.partition_datas_.push_back(partition);
  }
  return SUCCESS;
}

Status OmFileSaveHelper::SaveModel(const char_t *const output_file, ModelBufferData &model, const bool save_to_file,
                                   const bool is_partition_align, const uint32_t align_bytes) {
  if (model_contexts_.empty()) {
    GELOGE(FAILED, "mode contexts empty");
    return FAILED;
  }

  std::vector<ModelPartitionTable *> model_partition_tabels;
  std::vector<std::vector<ModelPartition>> all_model_partitions;
  for (size_t ctx_index = 0U; ctx_index < model_contexts_.size(); ++ctx_index) {
    auto &cur_ctx = model_contexts_[ctx_index];
    uint64_t cur_model_data_len = cur_ctx.model_data_len_;
    if (cur_model_data_len == 0U) {
      GELOGE(PARAM_INVALID, "Model data len error! should not be 0");
      return PARAM_INVALID;
    }

    // partition对齐
    if (is_partition_align) {
      cur_model_data_len = 0U;
      for (auto one_partition : cur_ctx.partition_datas_) {
        cur_model_data_len += MemSizeAlign(one_partition.size, align_bytes);
      }
    }
    ModelPartitionTable *const tmp_table = GetPartitionTable(ctx_index, is_partition_align, align_bytes);
    GE_CHECK_NOTNULL(tmp_table);
    uint64_t size_of_table = (SizeOfModelPartitionTable(*tmp_table));
    if (is_partition_align) {
      size_of_table = MemSizeAlign(size_of_table, align_bytes);
    }
    FMK_UINT64_ADDCHECK(size_of_table, cur_model_data_len)
    FMK_UINT64_ADDCHECK(size_of_table + cur_model_data_len, model_header_.model_length)

    model_header_.model_length += size_of_table + cur_model_data_len;
    model_partition_tabels.push_back(tmp_table);
    all_model_partitions.push_back(cur_ctx.partition_datas_);
    GELOGD("sizeof(ModelPartitionTable):%lu, cur_model_data_len:%lu, cur_context_index:%zu", size_of_table,
           cur_model_data_len, ctx_index);
  }
  Status ret;
  if (save_to_file) {
    ret = FileSaver::SaveToFile(
        output_file, model_header_, model_partition_tabels, all_model_partitions, is_partition_align, align_bytes);
  } else {
    ret = FileSaver::SaveToBuffWithFileHeader(model_header_, model_partition_tabels, all_model_partitions, model);
  }
  if (ret == SUCCESS) {
    GELOGD("Save model success.");
  }
  return ret;
}
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_ARGS_PARSER_FACTORY_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_ARGS_PARSER_FACTORY_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include "runtime/rt_model.h"
#include "aicpu_task_struct.h"
#include "proto/task.pb.h"
#include "graph/op_desc.h"
#include "graph/args_format_desc.h"
#include "ge/framework/common/taskdown_common.h"

#include "graph/load/model_manager/model_utils.h"

namespace ge {
class ArgsParser {
public:
  ArgsParser(const domi::TaskDef &task_def, OpDescPtr op_desc)
    : task_def_(task_def), op_desc_(std::move(op_desc)) {}
  virtual ~ArgsParser() = default;
  virtual uint64_t ParseArgsAddr(uint64_t args_addr, bool in, size_t io_index) const = 0;
  virtual bool CheckArgsExtra(uint64_t args_addr) const { return true; }
protected:
  const domi::TaskDef task_def_;
  const OpDescPtr op_desc_;
};

class ArgsParserFactory {
  using Creator = std::unique_ptr<ArgsParser> (*)(const domi::TaskDef &, OpDescPtr);
public:
  static std::unique_ptr<ArgsParser> CreateBy(const domi::TaskDef &task_def, OpDescPtr op_desc) {
    auto iter = map_.find(static_cast<ModelTaskType>(task_def.type()));
    if (iter == map_.end()) {
      return nullptr;
    }
    return iter->second(task_def, std::move(op_desc));
  }
  class Registerar {
    Registerar(ModelTaskType type, Creator creator) {
      ArgsParserFactory::map_.emplace(type, creator);
    }
  };
private:
  inline static std::unordered_map<ModelTaskType, Creator> map_;
};

#define REGISTER_PARSER_CREATOR(type, T)                                                        \
  using ArgsParser::ArgsParser;                                                                 \
  inline static ArgsParserFactory::Registerar registerar_{                                      \
    type,                                                                                       \
    [](const domi::TaskDef &task_def, OpDescPtr op_desc) -> std::unique_ptr<ArgsParser> {       \
      return std::make_unique<T>(task_def, std::move(op_desc));                                 \
    },                                                                                          \
  }

class KernelArgsParser : public ArgsParser {
public:
  uint64_t ParseArgsAddr(uint64_t args_addr, bool in, size_t io_index) const override {
    size_t offset = in ? io_index * sizeof(uint64_t)
                       : (ModelUtils::GetInputSize(op_desc_).size() + io_index) * sizeof(uint64_t);

    if (task_def_.kernel().context().kernel_type() == static_cast<uint32_t>(ccKernelType::AI_CPU)) {
      offset += sizeof(aicpu::AicpuParamHead);
    }
    return *reinterpret_cast<uint64_t *>(args_addr + offset);
  }
private:
  REGISTER_PARSER_CREATOR(ModelTaskType::MODEL_TASK_KERNEL, KernelArgsParser);
};

class CustomKernelArgsParser : public ArgsParser {
public:
  uint64_t ParseArgsAddr(uint64_t args_addr, bool in, size_t io_index) const override {
    size_t offset = in ? io_index * sizeof(uint64_t)
                       : (ModelUtils::GetInputSize(op_desc_).size() + io_index) * sizeof(uint64_t);

    if (task_def_.kernel().context().kernel_type() == static_cast<uint32_t>(ccKernelType::AI_CPU)) {
      offset += sizeof(aicpu::AicpuParamHead);
    }
    return *reinterpret_cast<uint64_t *>(args_addr + offset);
  }
private:
  REGISTER_PARSER_CREATOR(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL, CustomKernelArgsParser);
};

class DsaTaskArgsParser : public ArgsParser {
public:
  uint64_t ParseArgsAddr(uint64_t args_addr, bool in, size_t io_index) const override {
    const uint32_t *args = reinterpret_cast<uint32_t *>(args_addr);
    uint32_t addr_low, addr_high;
    if (in) {
      if (io_index == 0) { // cout
        addr_low = args[8];
        addr_high = args[9];
      } else if (io_index == 1) { // seed
        addr_low = args[6];
        addr_high = args[7];
      } else { // XXX: Other inputs are not parsed yet.
        addr_low = addr_high = 0;
      }
    } else {
      addr_low = args[0];
      addr_high = args[1];
    }
    return (static_cast<uint64_t>(addr_high) << 32) | addr_low;
  }
private:
  REGISTER_PARSER_CREATOR(ModelTaskType::MODEL_TASK_DSA, DsaTaskArgsParser);
};

class MemcpyAsyncArgsParser : public ArgsParser {
public:
  uint64_t ParseArgsAddr(uint64_t args_addr, bool in, size_t io_index) const override {
    const uint64_t *args = reinterpret_cast<uint64_t *>(args_addr);
    return in ? args[0] : args[1];
  }
private:
  REGISTER_PARSER_CREATOR(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC, MemcpyAsyncArgsParser);
};

class StreamSwitchArgsParser : public ArgsParser {
public:
  uint64_t ParseArgsAddr(uint64_t args_addr, bool in, size_t io_index) const override {
    const uint64_t *args = reinterpret_cast<uint64_t *>(args_addr);
    return args[io_index];
  }
private:
  REGISTER_PARSER_CREATOR(ModelTaskType::MODEL_TASK_STREAM_SWITCH, StreamSwitchArgsParser);
};

class MemcpyAddrAsyncArgsParser : public ArgsParser {
public:
  MemcpyAddrAsyncArgsParser(const domi::TaskDef &task_def, OpDescPtr op_desc)
    : ArgsParser(task_def, std::move(op_desc)) {
    const auto &format_str = task_def_.memcpy_async().args_format();
    if (ArgsFormatDesc::FromString(format_, op_desc_, format_str) != GRAPH_SUCCESS) {
      throw std::invalid_argument("Invalid args_format string");
    }
  }
  uint64_t ParseArgsAddr(uint64_t args_addr, bool in, size_t io_index) const override {
    size_t offset = 0;
    for (const auto &iter : format_) {
      if ((iter.addr_type == AddrType::INPUT_INSTANCE && in) ||
          (iter.addr_type == AddrType::OUTPUT_INSTANCE && !in)) {
        return *reinterpret_cast<uint64_t *>(args_addr + offset);
      }
      ArgsFormatDesc::GetArgSize(op_desc_, iter, offset);
    }
    GELOGW("%s not found in args_format[%s]", in ? "Input" : "Output", format_.ToString().c_str());
    return 0;
  }
  bool CheckArgsExtra(uint64_t args_addr) const override {
    const uint8_t *u8 = reinterpret_cast<uint8_t *>(args_addr);
    const uint64_t magic = 0xdeadbeef'deadbeef; // See rtMemcpyAsyncPtr.
    size_t offset = 0;
    for (const auto &iter : format_) {
      if (iter.addr_type == AddrType::PLACEHOLDER &&
          memcmp(&u8[offset], &magic, iter.ir_idx == static_cast<int32_t>(ArgsFormatWidth::BIT32) ?
                                      sizeof(uint32_t) : sizeof(uint64_t)) != 0) {
        GELOGE(FAILED, "placeholder args check failed at offset[%zu]", offset);
        return false;
      } else if (iter.addr_type == AddrType::CUSTOM_VALUE &&
                 memcmp(&u8[offset], iter.reserved, iter.ir_idx == static_cast<int32_t>(ArgsFormatWidth::BIT32) ?
                                                    sizeof(uint32_t) : sizeof(uint64_t)) != 0) {
        GELOGE(FAILED, "custom_value args check failed at offset[%zu]", offset);
        return false;
      }
      ArgsFormatDesc::GetArgSize(op_desc_, iter, offset);
    }
    GELOGD("CheckArgsExtra for MemcpyAddrAsync success, op[%s]", op_desc_->GetNamePtr());
    return true;
  }
private:
  REGISTER_PARSER_CREATOR(ModelTaskType::MODEL_TASK_MEMCPY_ADDR_ASYNC, MemcpyAddrAsyncArgsParser);
  ArgsFormatDesc format_;
};

} // namespace ge
#endif

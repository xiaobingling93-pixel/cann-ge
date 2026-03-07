/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/proto_util/proto_util.h"

#include "base/err_msg.h"

#include <fstream>

#include "mmpa/mmpa_api.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "graph/def_types.h"
#include "framework/common/util.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "common/checker.h"

namespace ge {
namespace {
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;

/*
 * kProtoReadBytesLimit and kWarningThreshold are real arguments of CodedInputStream::SetTotalBytesLimit.
 * In order to prevent integer overflow and excessive memory allocation during protobuf processing,
 * it is necessary to limit the length of proto message (call SetTotalBytesLimit function).
 * In theory, the minimum message length that causes an integer overflow is 512MB, and the default is 64MB.
 * If the limit of warning_threshold is exceeded, the exception information will be printed in stderr.
 * If such an exception is encountered during operation,
 * the proto file can be divided into several small files or the limit value can be increased.
 */
constexpr int32_t kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.
constexpr size_t kMaxErrorStrLength = 128U;
}  // namespace

static bool ReadProtoFromCodedInputStream(CodedInputStream &coded_stream, google::protobuf::Message *const proto) {
  if (proto == nullptr) {
    GELOGE(FAILED, "incorrect parameter. nullptr == proto");
    return false;
  }

  coded_stream.SetTotalBytesLimit(kProtoReadBytesLimit);
  return proto->ParseFromCodedStream(&coded_stream);
}

bool ReadProtoFromArray(const void *const data, const int32_t size, google::protobuf::Message *const proto) {
  if ((proto == nullptr) || (data == nullptr) || (size == 0)) {
    GELOGE(FAILED, "incorrect parameter. proto is nullptr || data is nullptr || size is 0");
    return false;
  }

  google::protobuf::io::CodedInputStream coded_stream(PtrToPtr<void, uint8_t>(const_cast<void *>(data)), size);
  return ReadProtoFromCodedInputStream(coded_stream, proto);
}

bool ReadProtoFromText(const char_t *const file, google::protobuf::Message *const message) {
  FuncPerfScope func_perf_scope(__FUNCTION__, file);
  if ((file == nullptr) || (message == nullptr)) {
    GELOGE(FAILED, "incorrect parameter. nullptr == file || nullptr == message");
    return false;
  }

  const std::string real_path = RealPath(file);
  GE_ASSERT_TRUE(!real_path.empty(), "[Do][RealPath]Path[%s]'s realpath is empty", file);

  if (GetFileLength(real_path) == -1) {
    GELOGE(FAILED, "file size not valid.");
    return false;
  }

  std::ifstream fs(real_path.c_str(), std::ifstream::in);
  if (!fs.is_open()) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const std::string errmsg = "[Errno " + std::to_string(mmGetErrorCode()) + "] " +
                               mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E13001",
        std::vector<const char *>({"file", "errmsg"}),
        std::vector<const char *>({real_path.c_str(), errmsg.c_str()})
    );
    GELOGE(ge::FAILED, "[Open][ProtoFile]Failed, real path %s, orginal file path %s",
           real_path.c_str(), file);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  const bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_ASSERT_TRUE(ret, "[Parse][File]Through [google::protobuf::TextFormat::Parse] failed, file %s", file);
  fs.close();

  return ret;
}
}  //  namespace ge

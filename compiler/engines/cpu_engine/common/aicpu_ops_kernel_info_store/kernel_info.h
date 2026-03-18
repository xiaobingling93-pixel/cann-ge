/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNEL_INFO_H_
#define AICPU_KERNEL_INFO_H_

#include <nlohmann/json.hpp>

#include "aicpu_ops_kernel_info_store/op_struct.h"
#include "aicpu_ops_kernel_info_store/aicpu_ops_kernel_info_store.h"
#include "error_code/error_code.h"
#include "factory/factory.h"

namespace aicpu {

class KernelInfo {
 public:
  /**
   * Destructor
   */
  virtual ~KernelInfo() = default;

  /**
   * Initialize related kernelinfo store
   * @param options configuration information
   * @return status whether this operation success
   */
  virtual ge::Status Initialize(
      const std::map<std::string, std::string> &options);

  /**
   * Release related resources of the aicpu kernelinfo store
   * @return status whether this operation success
   */
  ge::Status Finalize();

  /**
   * Member access method.
   * @param a map<string, OpInfo> stores operator's name and OpInfo
   * @return status whether this operation success
   */
  ge::Status GetOpInfos(
      std::map<std::string, aicpu::OpFullInfo> &op_infos) const;

  ge::Status GetOpInfo(const std::string &op_type, OpFullInfo &op_info) const;

  ge::Status GetCustUserInfo(std::map<std::string, std::string> &cust_user_info) const;

  virtual bool IsSupportedOps([[maybe_unused]] const std::string &op) const {
    return false;
  };

  /**
   * For ops in AIcore, Call CompileOp before Generate task in AICPU
   * @param node Node information
   * @return status whether operation successful
   */
  virtual ge::Status CompileOp(ge::NodePtr &node);

  bool has_cust_op_ = false;

  void SetJsonPath(const std::string &json_path);

  const std::string GetJsonPath();
 protected:
  /**
   * Read json file
   * @return whether read file successfully
   */
  virtual bool ReadOpInfoFromJsonFile() { return true; }

  const std::string GetOpsPath(const void *instance) const;

 protected:
  // kernelinfo json serialized object
  nlohmann::json op_info_json_file_;

  // 自定义算子包信息库
  std::vector<std::pair<std::string, nlohmann::json>> custop_info_json_file_;

  // store operator's name and detailed information
  std::map<std::string, aicpu::OpFullInfo> infos_;

  std::map<std::string, std::string> cust_user_infos_;

 private:
  ge::Status FillOpInfos(OpInfoDescs &info_desc);
  ge::Status GetCustOpInfo();
  ge::Status FillCustOpInfos(string user_name, OpInfoDescs &info_desc);
  std::string json_path_ = "";
};

#define FACTORY_KERNELINFO Factory<KernelInfo>

#define FACTORY_KERNELINFO_CLASS_KEY(CLASS, KEY) \
  FACTORY_KERNELINFO::Register<CLASS> __##CLASS(KEY);

}  // namespace aicpu

#endif  // AICPU_KERNEL_INFO_H_

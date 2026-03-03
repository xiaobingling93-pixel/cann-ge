/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_UDF_MODEL_BUILDER_H
#define UDF_UDF_MODEL_BUILDER_H

#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include "proto/udf_def.pb.h"
#include "udf_model.h"

namespace ge {
using UdfAttrMap = google::protobuf::Map<std::string, udf::AttrValue>;

class UdfModelBuilder {
 public:
  static UdfModelBuilder &GetInstance() {
    static UdfModelBuilder instance;
    return instance;
  }

  Status Build(UdfModel &udf_model) const;

 private:
  Status BuildFlowFuncOp(const OpDescPtr &op_desc, const bool is_heavy_load, const ComputeGraphPtr &graph,
                         const std::vector<CompileConfigJson::BufCfg> &buffer_configs, UdfModel &udf_model) const;

  Status SetDeployResource(const OpDescPtr &op_desc, UdfModel &udf_model, bool is_heavy_load,
                           const std::string &resource_type) const;

  Status BuildUdfDef(const OpDescPtr &op_desc, udf::UdfDef &udf_def) const;

  Status SetBin(const OpDescPtr &op_desc, udf::UdfDef &udf_def) const;

  Status SetBinName(const OpDescPtr &op_desc, udf::UdfDef &udf_def) const;

  Status SetFuncNameAndInputOutputMaps(const OpDescPtr &op_desc, udf::UdfDef &udf_def) const;

  Status GenReleasePackageForUserDefineFunc(UdfModel &udf_model, const OpDescPtr &op_desc,
      const std::string &resource_type, const ComputeGraphPtr &graph) const;

  Status GetAndCheckAttrs(const OpDescPtr &op_desc, const ComputeGraphPtr &graph,
      std::string &release_pkg_path, std::string &cache_release_info, std::string &om_model_file) const;

  Status GenReleasePackage(UdfModel &udf_model, const OpDescPtr &op_desc, const std::string &resource_type,
      const ComputeGraphPtr &graph) const;

  Status SetAttr(const std::string &attr_name, const AnyValue &value, UdfAttrMap &udf_attrs) const;

  Status SetStreamInputFuncNames(const std::string &op_name, const std::vector<NamedAttrs> &funcs_attr,
                                 udf::UdfDef &udf_def) const;

  Status SetMultiFuncInputOutputMaps(const std::string &op_name, const std::vector<NamedAttrs> &funcs_attr,
                                     udf::UdfDef &udf_def) const;
  Status SaveModelToFile(UdfModel &udf_model, const std::string &release_pkg_path,
                         const std::string &normalize_name) const;

  static Status GetReleaseInfo(const std::string &release_path, std::string &release_info);

  static Status PackRelease(const std::string &release_pkg_path, const std::string &resource_type,
                            const std::string &normalize_name);
  static void GenerateTarCmd(const std::string &release_pkg_path, const std::string &normalize_name,
                             const bool with_hash, std::string &pack_cmd);
  static Status PackReleaseWithHash(const std::string &release_pkg_path, const std::string &normalize_name);
  static Status PackReleaseWithoutHash(const std::string &release_pkg_path, const std::string &normalize_name);
  std::string GenNormalizeModelName(const std::string &model_name) const;
  Status ProcessForCache(UdfModel &udf_model, const std::string &om_model_file) const;
 private:
  UdfModelBuilder() = default;
  ~UdfModelBuilder() = default;
  UdfModelBuilder(const UdfModelBuilder &) = delete;
  UdfModelBuilder &operator=(const UdfModelBuilder &) = delete;
  UdfModelBuilder(UdfModelBuilder &&) = delete;
  UdfModelBuilder &operator=(UdfModelBuilder &&) = delete;
};
}  // namespace ge

#endif  // UDF_UDF_MODEL_BUILDER_H

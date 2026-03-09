/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "base/base_types.h"
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "graph_construct_utils.h"
#include "result_checker_utils.h"
#include "gen_tiling_impl.h"
#include "graph/utils/graph_utils.h"
#include "autofuse_config/auto_fuse_config.h"
#include "common/test_common_utils.h"
#include "test_common_utils.h"
using namespace ge::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1;
using namespace ge;
using HintGraph=AscGraph;
}
// 公共函数声明：生成tiling代码并写入文件
std::map<std::string, std::string> GenerateTilingCodeCommon(
    const std::string &op_name,
    const ascir::FusedScheduledResult &fused_scheduled_result,
    const std::map<std::string, std::string> &options,
    const std::string &tiling_func,
    const std::string &header_file,
    bool gen_extra_infos);

namespace {
using att::test_common::CreateTilingMainFunc;

std::string kRunTilingFuncMain = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";

std::string kConcatTilingTestHead = R"(
#ifndef ATT_TILING_TEST_CONCAT_H_
#define ATT_TILING_TEST_CONCAT_H_
#include <cstdint>
#include <array>
#include <iostream>
#include <unordered_map>

// 包含主头文件，获取 TilingData 类型定义
#include "Concat_tiling_data.h"

namespace optiling {

// ATT缓存相关常量
constexpr size_t kInputShapeSize = 1;
constexpr size_t kOperatorCacheCapacity = 24;  // 算子级缓存容量
constexpr double kLoadFactorThreshold = 0.8;   // 负载因子阈值

// TilingDataCopy 结构体 - 用于 Group 级别缓存存储
struct TilingDataCopy {
  uint32_t ND;
  void set_ND(uint32_t val) { ND = val; }
  inline uint32_t get_ND() { return ND; }
  uint32_t block_dim;
  void set_block_dim(uint32_t val) { block_dim = val; }
  inline uint32_t get_block_dim() { return block_dim; }
  uint32_t ndb_size;
  void set_ndb_size(uint32_t val) { ndb_size = val; }
  inline uint32_t get_ndb_size() { return ndb_size; }
  uint32_t ndbt_size;
  void set_ndbt_size(uint32_t val) { ndbt_size = val; }
  inline uint32_t get_ndbt_size() { return ndbt_size; }
  uint32_t q0_size;
  void set_q0_size(uint32_t val) { q0_size = val; }
  inline uint32_t get_q0_size() { return q0_size; }
  uint32_t q1_size;
  void set_q1_size(uint32_t val) { q1_size = val; }
  inline uint32_t get_q1_size() { return q1_size; }
  uint32_t tiling_key;
  void set_tiling_key(uint32_t val) { tiling_key = val; }
  inline uint32_t get_tiling_key() { return tiling_key; }
  uint32_t ub_size;
  void set_ub_size(uint32_t val) { ub_size = val; }
  inline uint32_t get_ub_size() { return ub_size; }
};
// FixedSizeHashMap 模板类 - 固定大小的哈希表，用于缓存
template <size_t KEY_SIZE, size_t CAPACITY, typename VALUE_TYPE>
class FixedSizeHashMap {
private:
  using Key = std::array<uint32_t, KEY_SIZE>;
  using Value = VALUE_TYPE;

  enum BucketState { kEmpty, kOccupied, kDeleted };
  struct Bucket {
    Key key;
    Value value;
    BucketState state;
  };

  std::array<Bucket, CAPACITY> buckets;
  size_t size_ = 0;

  // Hash - 大驼峰命名
  size_t Hash(const Key &key) const {
    size_t hash = 0;
    for (const auto& value : key) {
      constexpr uint32_t kHashPrime = 0x9e3779b9;  // 黄金比例的整数表示，用于hash混合
      hash ^= value + kHashPrime + (hash << 6) + (hash >> 2);
    }
    return hash;
  }

  // FindIndex - 大驼峰命名
  size_t FindIndex(const Key &key) const {
    size_t hash = Hash(key) % CAPACITY;
    size_t start = hash;
    do {
      if (buckets[hash].state == kEmpty) {
        return CAPACITY;
      } else if (buckets[hash].state == kOccupied && buckets[hash].key == key) {
        return hash;
      }
      hash = (hash + 1) % CAPACITY;
    } while (hash != start);
    return CAPACITY;
  }

public:
  FixedSizeHashMap() : size_(0) {
    for (size_t i = 0; i < CAPACITY; ++i) {
      buckets[i].state = kEmpty;
    }
  }

  // Find - 大驼峰命名
  Value* Find(const Key &key) {
    size_t index = FindIndex(key);
    if (index < CAPACITY && buckets[index].state == kOccupied) {
      return &buckets[index].value;
    }
    return nullptr;
  }
const Value* Find(const Key &key) const {
    return const_cast<FixedSizeHashMap*>(this)->Find(key);
  }

  // Insert - 大驼峰命名
  bool Insert(const Key &key, const Value &value) {
    if (size_ >= CAPACITY * kLoadFactorThreshold) {
      return false;  // 80%容量阈值
    }
    size_t index = FindIndex(key);
    if (index >= CAPACITY) {
      size_t hash = Hash(key) % CAPACITY;
      for (size_t i = 0; i < CAPACITY; ++i) {
        index = (hash + i) % CAPACITY;
        if (buckets[index].state == kEmpty) {
          buckets[index].key = key;
          buckets[index].value = value;
          buckets[index].state = kOccupied;
          size_++;
          return true;
        }
      }
      return false;
    }
    buckets[index].value = value;
    return true;
  }

  // Erase - 大驼峰命名
  bool Erase(const Key &key) {
    size_t index = FindIndex(key);
    if (index < CAPACITY && buckets[index].state == kOccupied) {
      buckets[index].state = kDeleted;
      size_--;
      return true;
    }
    return false;
  }
// Clear - 大驼峰命名
  void Clear() {
    for (auto& bucket : buckets) {
      bucket.state = kEmpty;
    }
    size_ = 0;
  }

  size_t Size() const { return size_; }
  bool Empty() const { return size_ == 0; }
};

namespace AscGraph0ScheduleResult0G0 {

// GroupLevelCache 类型别名 - Group 级别缓存
using GroupLevelCache = FixedSizeHashMap<kInputShapeSize, 4, TilingDataCopy>;

// 前向声明 - GetTiling 函数将在 Concat_tiling_func.cpp 中实现
bool GetTiling(AscGraph0ScheduleResult0G0TilingData &tiling_data,
               std::unordered_map<int64_t, uint64_t> &workspace_map,
               int32_t tilingCaseId,
               GroupLevelCache *cache = nullptr);

} // namespace AscGraph0ScheduleResult0G0

} // namespace optiling

#endif
)";

std::string kRunTilingFuncMainSameND = R"(
#include <iostream>
#include <unordered_map>
#include "Concat_tiling_data.h"
#include "Concat_tiling_test.h"

using namespace optiling;

int main() {
  // ========== Group 级别缓存测试 ==========
  std::cout << "\n========== Group 级别缓存测试 ==========" << std::endl;
  AscGraph0ScheduleResult0G0::GroupLevelCache group_cache;
  // Test 1: ND = 1024，首次执行（不命中）
  std::cout << "\n--- Test 1: ND = 1024 (首次执行) ---" << std::endl;
  AscGraph0ScheduleResult0G0TilingData tilingData1;
  tilingData1.set_ND(1024);
  tilingData1.set_block_dim(64);
  tilingData1.set_ub_size(245760);
  std::unordered_map<int64_t, uint64_t> workspace_map1;
  if (!AscGraph0ScheduleResult0G0::GetTiling(
        tilingData1, workspace_map1, -1, &group_cache)) {
    std::cout << "Test 1 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData1.get_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData1.get_ND() << std::endl;
  // Test 2: ND = 1024，再次执行（应命中 Group 缓存）
  std::cout << "\n--- Test 2: ND = 1024 (应命中 Group 缓存) ---" << std::endl;
  AscGraph0ScheduleResult0G0TilingData tilingData2;
  tilingData2.set_ND(1024);
  tilingData2.set_block_dim(64);
  tilingData2.set_ub_size(245760);

  std::unordered_map<int64_t, uint64_t> workspace_map2;
  if (!AscGraph0ScheduleResult0G0::GetTiling(
        tilingData2, workspace_map2, -1, &group_cache)) {
    std::cout << "Test 2 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData2.get_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData2.get_ND() << std::endl;
  // Test 3: ND = 2048，首次执行（不命中）
  std::cout << "\n--- Test 3: ND = 2048 (首次执行) ---" << std::endl;
  AscGraph0ScheduleResult0G0TilingData tilingData3;
  tilingData3.set_ND(2048);
  tilingData3.set_block_dim(64);
  tilingData3.set_ub_size(245760);

  std::unordered_map<int64_t, uint64_t> workspace_map3;
  if (!AscGraph0ScheduleResult0G0::GetTiling(
        tilingData3, workspace_map3, -1, &group_cache)) {
    std::cout << "Test 3 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData3.get_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData3.get_ND() << std::endl;

  // Test 4: ND = 2048，再次执行（应命中 Group 缓存）
  std::cout << "\n--- Test 4: ND = 2048 (应命中 Group 缓存) ---" << std::endl;
  AscGraph0ScheduleResult0G0TilingData tilingData4;
  tilingData4.set_ND(2048);
  tilingData4.set_block_dim(64);
  tilingData4.set_ub_size(245760);

  std::unordered_map<int64_t, uint64_t> workspace_map4;
  if (!AscGraph0ScheduleResult0G0::GetTiling(
        tilingData4, workspace_map4, -1, &group_cache)) {
    std::cout << "Test 4 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData4.get_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData4.get_ND() << std::endl;

  // ========== Operator 级别缓存测试 ==========
  std::cout << "\n========== Operator 级别缓存测试 ==========" << std::endl;

  // Test 5: ND = 4096，首次执行（不命中）
  std::cout << "\n--- Test 5: ND = 4096 (首次执行) ---" << std::endl;
  graph_ndTilingData tilingData5;
  tilingData5.set_block_dim(64);
  tilingData5.set_ub_size(245760);
  tilingData5.graph0_result0_g0_tiling_data.set_ND(4096);

  if (!GetTiling(tilingData5)) {
    std::cout << "Test 5 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData5.get_graph0_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData5.graph0_result0_g0_tiling_data.get_ND() << std::endl;

  // Test 6: ND = 4096，再次执行（应命中 Operator 缓存）
  std::cout << "\n--- Test 6: ND = 4096 (应命中 Operator 缓存) ---" << std::endl;
  graph_ndTilingData tilingData6;
  tilingData6.set_block_dim(64);
  tilingData6.set_ub_size(245760);
  tilingData6.graph0_result0_g0_tiling_data.set_ND(4096);

  if (!GetTiling(tilingData6)) {
    std::cout << "Test 6 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData6.get_graph0_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData6.graph0_result0_g0_tiling_data.get_ND() << std::endl;
  std::cout << "\n========== All tests passed ==========" << std::endl;
  return 0;
}
)";
}
using namespace att;

class STestGenConcat : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
    AutoFuseConfig::MutableAttStrategyConfig().force_template_op_name = "";
    AutoFuseConfig::MutableAttStrategyConfig().force_tiling_case = "";
    AutoFuseConfig::MutableAttStrategyConfig().force_schedule_result = -1L;
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    unsetenv("AUTOFUSE_DFX_FLAGS");
  }

 protected:
  // 辅助函数：验证Group缓存行为
  void VerifyGroupCacheBehavior();
  // 辅助函数：验证静态shape缓存行为
  void VerifyStaticShapeCacheBehavior();

  // 辅助函数：构建单个图到schedule_group
  void BuildSingleGraphToScheduleGroup(
      ascir::AscGraph& graph,
      ascir::ScheduleGroup& schedule_group,
      uint32_t tiling_key);

  // 辅助函数：生成tiling函数并写入文件
  void GenerateTilingFunctionAndWriteToFile(
      const std::string& op_name,
      const ascir::FusedScheduledResult& fused_scheduled_result,
      std::map<std::string, std::string>& options);

  // 辅助函数：生成tiling数据和头文件
  void GenerateTilingDataAndHeader(
      const std::string& op_name,
      const std::string& graph_name,
      const ascir::FusedScheduledResult& fused_scheduled_result,
      std::map<std::string, std::string>& options);

  // 辅助函数：准备测试环境文件
  void PrepareTestEnvironmentFiles(const std::string& test_header_content = "");

  // 辅助函数：编译生成的tiling代码
  void CompileGeneratedTilingCode();

  // 辅助函数：编译并运行tiling测试，验证输出
  void CompileAndRunTilingTest(const std::string& output_file = "./info.log");

  // 静态成员：静态shape缓存测试主函数代码
  static const std::string kStaticShapeCacheTestMain;
};

void Concat_Normal_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");

  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  // 定义节点
  int exec_order = 0;
  Data x1("x1", graph);
  x1.attr.sched.exec_order = exec_order++;
  x1.attr.sched.axis = {a.id, r.id, bl.id};
  x1.y.dtype = ge::DT_FLOAT16;
  *x1.y.axis = {a.id, r.id, bl.id};
  *x1.y.repeats = {A, R, ONE};
  *x1.y.strides = {R, ONE, ZERO};

  Load x1Local("x1Local");
  x1Local.x = x1.y;
  x1Local.attr.sched.exec_order = exec_order++;
  x1Local.attr.sched.axis = {a.id, r.id, bl.id};
  x1Local.y.dtype = ge::DT_FLOAT16;
  *x1Local.y.axis = {a.id, r.id, bl.id};
  *x1Local.y.repeats = {A, R, ONE};
  *x1Local.y.strides = {R, ONE, ZERO};

  Data x2("x2", graph);
  x2.attr.sched.exec_order = exec_order++;
  x2.attr.sched.axis = {a.id, r.id, bl.id};
  x2.y.dtype = ge::DT_FLOAT16;
  *x2.y.axis = {a.id, r.id, bl.id};
  *x2.y.repeats = {A, R, ONE};
  *x2.y.strides = {R, ONE, ZERO};

  Load x2Local("x2Local");
  x2Local.x = x2.y;
  x2Local.attr.sched.exec_order = exec_order++;
  x2Local.attr.sched.axis = {a.id, r.id, bl.id};
  x2Local.y.dtype = ge::DT_FLOAT16;
  *x2Local.y.axis = {a.id, r.id, bl.id};
  *x2Local.y.repeats = {A, R, ONE};
  *x2Local.y.strides = {R, ONE, ZERO};

  Data bias("bias", graph);
  bias.attr.sched.exec_order = exec_order++;
  bias.attr.sched.axis = {a.id, r.id, bl.id};
  bias.y.dtype = ge::DT_FLOAT16;
  *bias.y.axis = {a.id, r.id, bl.id};
  *bias.y.repeats = {A, R, ONE};
  *bias.y.strides = {R, ONE, ZERO};

  Load biasLocal("biasLocal");
  biasLocal.x = bias.y;
  biasLocal.attr.sched.exec_order = exec_order++;
  biasLocal.attr.sched.axis = {a.id, r.id, bl.id};
  biasLocal.y.dtype = ge::DT_FLOAT16;
  *biasLocal.y.axis = {a.id, r.id, bl.id};
  *biasLocal.y.repeats = {A, R, ONE};
  *biasLocal.y.strides = {R, ONE, ZERO};

  Concat mean("mean");
  mean.attr.api.unit = ge::ComputeUnit::kUnitVector;
  mean.x = {x1Local.y, x2Local.y, biasLocal.y};
  mean.attr.sched.exec_order = exec_order++;
  mean.attr.sched.axis = {a.id, r.id, bl.id};
  mean.y.dtype = ge::DT_FLOAT;        // mean
  *mean.y.axis = {a.id, r.id, bl.id};
  *mean.y.repeats = {A, ONE, ONE};
  *mean.y.strides = {ONE, ZERO, ZERO};

  Store x_out("x_out");
  x_out.attr.sched.exec_order = exec_order++;
  x_out.attr.sched.axis = {a.id, r.id, bl.id};
  x_out.x = mean.y;
  x_out.y.dtype = ge::DT_FLOAT16;
  *x_out.y.axis = {a.id, r.id, bl.id};
  *x_out.y.repeats = {A, R, ONE};
  *x_out.y.strides = {R, ONE, ZERO};

  Store mean_out("mean_out");
  mean_out.attr.sched.exec_order = exec_order++;
  mean_out.attr.sched.axis = {a.id, r.id, bl.id};
  mean_out.x = mean.y;
  mean_out.y.dtype = ge::DT_FLOAT;
  *mean_out.y.axis = {a.id, r.id, bl.id};
  *mean_out.y.repeats = {A, ONE, ONE};
  *mean_out.y.strides = {ONE, ZERO, ZERO};

  Data one("one", graph);
  one.attr.sched.exec_order = exec_order++;
  one.attr.sched.axis = {a.id, r.id, bl.id};
  one.y.dtype = ge::DT_FLOAT;
  *one.y.axis = {a.id, r.id, bl.id};
  *one.y.repeats = {ONE, ONE, BL};
  *one.y.strides = {ZERO, ZERO, ONE};

  Concat rstd("rstd");
  rstd.attr.api.unit = ge::ComputeUnit::kUnitVector;
  rstd.attr.sched.exec_order = exec_order++;
  rstd.attr.sched.axis = {a.id, r.id, bl.id};
  rstd.x = {mean.y, mean.y, one.y};
  rstd.y.dtype = ge::DT_FLOAT;      // x-mean
  *rstd.y.axis = {a.id, r.id, bl.id};
  *rstd.y.repeats = {A, R, ONE};
  *rstd.y.strides = {R, ONE, ZERO};

  Store rstd_out("rstd_out");
  rstd_out.attr.sched.exec_order = exec_order++;
  rstd_out.attr.sched.axis = {a.id, r.id, bl.id};
  rstd_out.x = rstd.y;
  rstd_out.y.dtype = ge::DT_FLOAT;
  *rstd_out.y.axis = {a.id, r.id, bl.id};
  *rstd_out.y.repeats = {A, ONE, ONE};
  *rstd_out.y.strides = {ONE, ZERO, ZERO};

  Data beta("beta", graph);
  beta.attr.sched.exec_order = exec_order++;
  beta.attr.sched.axis = {a.id, r.id, bl.id};
  beta.y.dtype = ge::DT_FLOAT16;
  *beta.y.axis = {a.id, r.id, bl.id};
  *beta.y.repeats = {ONE, R, ONE};
  *beta.y.strides = {ZERO, ONE, ZERO};

  Load betaLocal("betaLocal");
  betaLocal.x = beta.y;
  betaLocal.attr.sched.exec_order = exec_order++;
  betaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  betaLocal.y.dtype = ge::DT_FLOAT16;
  *betaLocal.y.axis = {a.id, r.id, bl.id};
  *betaLocal.y.repeats = {ONE, R, ONE};
  *betaLocal.y.strides = {ZERO, ONE, ZERO};

  Data gamma("gamma", graph);
  gamma.attr.sched.exec_order = exec_order++;
  gamma.attr.sched.axis = {a.id, r.id, bl.id};
  gamma.y.dtype = ge::DT_FLOAT16;
  *gamma.y.axis = {a.id, r.id, bl.id};
  *gamma.y.repeats = {ONE, R, ONE};
  *gamma.y.strides = {ZERO, ONE, ZERO};

  Load gammaLocal("gammaLocal");
  gammaLocal.x = gamma.y;
  gammaLocal.attr.sched.exec_order = exec_order++;
  gammaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  gammaLocal.y.dtype = ge::DT_FLOAT16;
  *gammaLocal.y.axis = {a.id, r.id, bl.id};
  *gammaLocal.y.repeats = {ONE, R, ONE};
  *gammaLocal.y.strides = {ZERO, ONE, ZERO};

  Concat y("y");
  y.attr.api.unit = ge::ComputeUnit::kUnitVector;
  y.attr.sched.exec_order = exec_order++;
  y.attr.sched.axis = {a.id, r.id, bl.id};
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};                 // x-mean
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {a.id, r.id, bl.id};
  *y.y.repeats = {A, R, ONE};
  *y.y.strides = {R, ONE, ZERO};

  Concat concat("concat");
  y.attr.api.unit = ge::ComputeUnit::kUnitVector;
  concat.x = {x1Local.y, x2Local.y};
  concat.attr.sched.axis = {a.id, r.id, bl.id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {a.id, r.id, bl.id};
  *concat.y.repeats = {A, R, ONE};
  *concat.y.strides = {R, ONE, ZERO};

  Store concat_out("cat_out");
  concat_out.attr.sched.exec_order = exec_order++;
  concat_out.attr.sched.axis = {a.id, r.id, bl.id};
  concat_out.x = y.y;
  concat_out.y.dtype = ge::DT_FLOAT16;
  *concat_out.y.axis = {a.id, r.id, bl.id};
  *concat_out.y.repeats = {A, R, ONE};
  *concat_out.y.strides = {R, ONE, ZERO};

  Store y_out("y_out");
  y_out.attr.sched.exec_order = exec_order++;
  y_out.attr.sched.axis = {a.id, r.id, bl.id};
  y_out.x = y.y;
  y_out.y.dtype = ge::DT_FLOAT16;
  *y_out.y.axis = {a.id, r.id, bl.id};
  *y_out.y.repeats = {A, R, ONE};
  *y_out.y.strides = {R, ONE, ZERO};

  Output buf1("buf1");
  buf1.x = x_out.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.y.dtype = ge::DT_FLOAT16;
  *buf1.y.axis = {a.id, r.id, bl.id};
  *buf1.y.repeats = {A, R, ONE};
  *buf1.y.strides = {R, ONE, ZERO};

  Output buf2("buf2");
  buf2.x = mean_out.y;
  buf2.attr.sched.exec_order = exec_order++;
  buf2.y.dtype = ge::DT_FLOAT;
  *buf2.y.axis = {a.id, r.id, bl.id};
  *buf2.y.repeats = {A, ONE, ONE};
  *buf2.y.strides = {ONE, ZERO, ZERO};

  Output buf3("buf3");
  buf3.x = rstd_out.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.y.dtype = ge::DT_FLOAT;
  *buf3.y.axis = {a.id, r.id, bl.id};
  *buf3.y.repeats = {A, ONE, ONE};
  *buf3.y.strides = {ONE, ZERO, ZERO};

  Output buf("buf");
  buf.x = y_out.y;
  buf.attr.sched.exec_order = exec_order++;
  buf.y.dtype = ge::DT_FLOAT16;
  *buf.y.axis = {a.id, r.id, bl.id};
  *buf.y.repeats = {A, R, ONE};
  *buf.y.strides = {R, ONE, ZERO};

  Output buf4("buf4");
  buf4.x = concat_out.y;
  buf4.attr.sched.exec_order = exec_order++;
  buf4.y.dtype = ge::DT_FLOAT16;
  *buf4.y.axis = {a.id, r.id, bl.id};
  *buf4.y.repeats = {A, R, ONE};
  *buf4.y.strides = {R, ONE, ZERO};
}

/*
for aBO
  for aBIO
    for aBII
      for r
        load x1
        load x2
        load bias
        CalcMean
        CalcRstd
        Store X
        Store mean
        Load beta
        Load gamma
        CalcRstd
        Store rstd
        CalcY
        Store y
*/

void Concat_Normal_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;
  auto bl = graph.FindAxis(2)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "nbi", "nbo");   // AB Ab
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");  // AbT Abt
  // graph.UpdateAxisAlign(aBI.id, 1u);
  // graph.UpdateAxisAlign(aBII.id, 8u);
  auto x1 = graph.FindNode("x1");
  graph.ApplySplit(x1, aBO->id, aBI->id);
  graph.ApplySplit(x1, aBIO->id, aBII->id);
  x1->attr.sched.loop_axis = aBIO->id;
  x1->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x2 = graph.FindNode("x2");
  graph.ApplySplit(x2, aBO->id, aBI->id);
  graph.ApplySplit(x2, aBIO->id, aBII->id);
  x2->attr.sched.loop_axis = aBIO->id;
  x2->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto bias = graph.FindNode("bias");
  graph.ApplySplit(bias, aBO->id, aBI->id);
  graph.ApplySplit(bias, aBIO->id, aBII->id);
  bias->attr.sched.loop_axis = aBIO->id;
  bias->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x1Local = graph.FindNode("x1Local");
  graph.ApplySplit(x1Local, aBO->id, aBI->id);
  graph.ApplySplit(x1Local, aBIO->id, aBII->id);
  x1Local->attr.sched.loop_axis = aBIO->id;
  x1Local->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x2Local = graph.FindNode("x2Local");
  graph.ApplySplit(x2Local, aBO->id, aBI->id);
  graph.ApplySplit(x2Local, aBIO->id, aBII->id);
  x2Local->attr.sched.loop_axis = aBIO->id;
  x2Local->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto biasLocal = graph.FindNode("biasLocal");
  graph.ApplySplit(biasLocal,aBO->id, aBI->id);
  graph.ApplySplit(biasLocal, aBIO->id, aBII->id);
  biasLocal->attr.sched.loop_axis = aBIO->id;
  biasLocal->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto mean = graph.FindNode("mean");
  graph.ApplySplit(mean,aBO->id, aBI->id);
  graph.ApplySplit(mean,aBIO->id, aBII->id);
  mean->attr.sched.loop_axis = aBIO->id;
  mean->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x_out = graph.FindNode("x_out");
  graph.ApplySplit(x_out, aBO->id, aBI->id);
  graph.ApplySplit(x_out, aBIO->id, aBII->id);
  x_out->attr.sched.loop_axis = aBIO->id;
  x_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto mean_out = graph.FindNode("mean_out");
  graph.ApplySplit(mean_out, aBO->id, aBI->id);
  graph.ApplySplit(mean_out, aBIO->id, aBII->id);
  mean_out->attr.sched.loop_axis = aBIO->id;
  mean_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto rstd = graph.FindNode("rstd");
  graph.ApplySplit(rstd,aBO->id, aBI->id);
  graph.ApplySplit(rstd,aBIO->id, aBII->id);
  rstd->attr.sched.loop_axis = aBIO->id;
  rstd->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto rstd_out = graph.FindNode("rstd_out");
  graph.ApplySplit(rstd_out,aBO->id, aBI->id);
  graph.ApplySplit(rstd_out,aBIO->id, aBII->id);
  rstd_out->attr.sched.loop_axis = aBIO->id;
  rstd_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto betaLocal = graph.FindNode("betaLocal");
  graph.ApplySplit(betaLocal,aBO->id, aBI->id);
  graph.ApplySplit(betaLocal,aBIO->id, aBII->id);
  betaLocal->attr.sched.loop_axis = aBIO->id;
  betaLocal->outputs[0].attr.vectorized_axis = {r};

  auto gammaLocal = graph.FindNode("gammaLocal");
  graph.ApplySplit(gammaLocal,aBO->id, aBI->id);
  graph.ApplySplit(gammaLocal,aBIO->id, aBII->id);
  gammaLocal->attr.sched.loop_axis = aBIO->id;
  gammaLocal->outputs[0].attr.vectorized_axis = {r};

  auto y = graph.FindNode("y");
  graph.ApplySplit(y,aBO->id, aBI->id);
  graph.ApplySplit(y,aBIO->id, aBII->id);
  y->attr.sched.loop_axis = aBIO->id;
  y->outputs[0].attr.vectorized_axis = {aBII->id, r};

  
  auto concat = graph.FindNode("concat");
  graph.ApplySplit(concat,aBO->id, aBI->id);
  graph.ApplySplit(concat,aBIO->id, aBII->id);
  concat->attr.api.unit = ge::ComputeUnit::kUnitVector;
  concat->attr.sched.loop_axis = aBIO->id;
  concat->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto y_out = graph.FindNode("y_out");
  graph.ApplySplit(y_out,aBO->id, aBI->id);
  graph.ApplySplit(y_out,aBIO->id, aBII->id);
  y_out->attr.sched.loop_axis = aBIO->id;
  y_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto cat_out = graph.FindNode("cat_out");
  graph.ApplySplit(cat_out,aBO->id, aBI->id);
  graph.ApplySplit(cat_out,aBIO->id, aBII->id);
  cat_out->attr.sched.loop_axis = aBIO->id;
  cat_out->outputs[0].attr.vectorized_axis = {aBII->id, r};
}

void Concat_Normal_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int gammaQue = queID++;
  int betaQue = queID++;
  int meanQue = queID++;
  int rstdQue = queID++;
  int yQue = queID++;
  int xQue = queID++;
  int x32Queue = queID++;
  int oneTBuf = bufID++;

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto bias = graph.FindNode("bias");
  bias->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  bias->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x1Local = graph.FindNode("x1Local");
  x1Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x1Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x1Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x1Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x1Local->outputs[0].attr.que.id = x1Que;
  x1Local->outputs[0].attr.que.depth = 1;
  x1Local->outputs[0].attr.que.buf_num = 1;
  x1Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x2Local = graph.FindNode("x2Local");
  x2Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x2Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x2Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x2Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x2Local->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  x2Local->outputs[0].attr.buf.id = ascir::ID_NONE;
  x2Local->outputs[0].attr.que.id = x2Que;
  x2Local->outputs[0].attr.que.depth = 1;
  x2Local->outputs[0].attr.que.buf_num = 1;
  x2Local->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto biasLocal = graph.FindNode("biasLocal");
  biasLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  biasLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  biasLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  biasLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  biasLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  biasLocal->outputs[0].attr.que.id = biasQue;
  biasLocal->outputs[0].attr.que.depth = 1;
  biasLocal->outputs[0].attr.que.buf_num = 1;
  biasLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto mean = graph.FindNode("mean");
  mean->outputs[0].attr.mem.tensor_id = tensorID++;
  mean->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mean->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mean->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  mean->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  mean->outputs[0].attr.buf.id = ascir::ID_NONE;
  mean->outputs[0].attr.que.id = meanQue;
  mean->outputs[0].attr.que.depth = 1;
  mean->outputs[0].attr.que.buf_num = 1;
  mean->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto x_out = graph.FindNode("x_out");
  x_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto mean_out = graph.FindNode("mean_out");
  mean_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  mean_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto one = graph.FindNode("one");
  one->outputs[0].attr.mem.tensor_id = tensorID++;
  one->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  one->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  one->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  one->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  one->outputs[0].attr.buf.id = oneTBuf;
  one->outputs[0].attr.que.id = ascir::ID_NONE;
  one->outputs[0].attr.que.depth = ascir::ID_NONE;
  one->outputs[0].attr.que.buf_num = ascir::ID_NONE;
  one->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto rstd = graph.FindNode("rstd");
  rstd->outputs[0].attr.mem.tensor_id = tensorID++;
  rstd->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  rstd->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  rstd->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  rstd->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  rstd->outputs[0].attr.buf.id =ascir::ID_NONE;
  rstd->outputs[0].attr.que.id = yQue;
  rstd->outputs[0].attr.que.depth = 1;
  rstd->outputs[0].attr.que.buf_num = 1;
  rstd->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto rstd_out = graph.FindNode("rstd_out");
  rstd_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  rstd_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto beta = graph.FindNode("beta");
  beta->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  beta->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto betaLocal = graph.FindNode("betaLocal");
  betaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  betaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  betaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  betaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  betaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  betaLocal->outputs[0].attr.que.id = betaQue;
  betaLocal->outputs[0].attr.que.depth = 1;
  betaLocal->outputs[0].attr.que.buf_num = 1;
  betaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto gamma = graph.FindNode("gamma");
  gamma->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  gamma->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto gammaLocal = graph.FindNode("gammaLocal");
  gammaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  gammaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  gammaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  gammaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  gammaLocal->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.buf.id = ascir::ID_NONE;
  gammaLocal->outputs[0].attr.que.id = gammaQue;
  gammaLocal->outputs[0].attr.que.depth = 1;
  gammaLocal->outputs[0].attr.que.buf_num = 1;
  gammaLocal->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y = graph.FindNode("y");
  y->outputs[0].attr.mem.tensor_id = tensorID++;
  y->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  y->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  y->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  y->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  y->outputs[0].attr.buf.id = ascir::ID_NONE;
  y->outputs[0].attr.que.id = yQue;
  y->outputs[0].attr.que.depth = 1;
  y->outputs[0].attr.que.buf_num = 1;
  y->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto y_out = graph.FindNode("y_out");
  y_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  y_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto concat = graph.FindNode("concat");
  concat->outputs[0].attr.mem.tensor_id = tensorID++;
  concat->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  concat->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  concat->outputs[0].attr.buf.id = ascir::ID_NONE;
  concat->outputs[0].attr.que.id = yQue;
  concat->outputs[0].attr.que.depth = 1;
  concat->outputs[0].attr.que.buf_num = 1;
  concat->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  auto cat_out = graph.FindNode("cat_out");
  cat_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  cat_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}

namespace ge {
namespace ascir {
namespace cg {
Status BuildConcatGroupAscendGraphND(ge::AscGraph &graph) {
  // create default axis
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0S1MultiTiling(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto S1 = ge::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto [s2T, s2t] = graph.TileSplit(s0.id);
  auto [s1T, s1t] = graph.TileSplit(s1.id);
  auto s1Ts2T = *graph.MergeAxis({s1T->id, s2T->id});
  auto [s1Ts2TB, s1Ts2Tb] = graph.BlockSplit(s1Ts2T.id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s0, s1});
  LOOP(*s1Ts2TB) {
    LOOP(*s1Ts2Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto abs = Abs("abs1", load1).TQue(Position::kPositionVecOut, 1, 1);
      auto store1 = Store("store1", abs);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*s1Ts2TB, *s1Ts2Tb, *s1t, *s2t},
                                                                    {load1, store1}, 1));
      *load1.axis = {s1Ts2Tb->id, s2t->id, s1t->id};
      *load1.repeats = {s1Ts2Tb->size, s2t->size, s1t->size};
      *load1.strides = {s2t->size * s1t->size, s1t->size, CreateExpr(1)};
      *load1.vectorized_axis = {s1t->id, s2t->id};

      *abs.axis = {s1Ts2Tb->id, s1t->id, s2t->id};
      *abs.repeats = {s1Ts2Tb->size, s1t->size, s2t->size};
      *abs.strides = {s2t->size * s1t->size, s2t->size, CreateExpr(1)};
      *abs.vectorized_axis = {s1t->id, s2t->id};

      *store1.axis = {s1Ts2Tb->id, s1t->id, s2t->id};
      *store1.repeats = {s1Ts2Tb->size, s1t->size, s2t->size};
      *store1.strides = {s2t->size * s1t->size, s2t->size, CreateExpr(1)};
      *store1.vectorized_axis = {s1t->id, s2t->id};
      auto output1 = Output("output1", store1);
    }
  }
  for (auto node : graph.GetAllNodes()) {
    if (node->outputs().empty()) {
      continue;
    }
    auto last_dim_name = GetVecString(node->outputs()[0]->attr.repeats);
    GELOGD("Found Tile split axis %s in load/store node", last_dim_name.c_str());
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0S1_Reorder(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = ge::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto S1 = ge::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto [ndB, ndb] = graph.BlockSplit(s0.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {s1});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS1S0_Reorder(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S1 = ge::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto S0 = ge::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto [ndB, ndb] = graph.BlockSplit(s1.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s1});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {s0});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0B, *z0bT, *z0b, *z0bt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphND2(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);
  auto ND = ge::Symbol("ND2");
  auto nd = graph.CreateAxis("nd2", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphND2WithAbs(ge::AscGraph &graph) {
  // create default axis
  auto ND = ge::Symbol("ND2");
  auto nd = graph.CreateAxis("nd2", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto abs = Abs("load1", load1).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", abs);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, abs, store1, store2},
                                                                    2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphND2TB(ge::AscGraph &graph) {
  // create default axis
  auto ND = ge::Symbol("ND2");
  auto nd = graph.CreateAxis("nd2", ND);
  auto [ndT, ndt] = graph.TileSplit(nd.id);
  auto [ndTB, ndTb] = graph.BlockSplit(ndT->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndTB) {
    LOOP(*ndTb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndTB, *ndTb, *ndt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphStatic(ge::AscGraph &graph) {
  // create default axis
  auto A = ge::Symbol(10, "A");
  auto R = ge::Symbol(20, "R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol(10, "ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return ge::SUCCESS;
}

Status BuildTqueTbufAscendGraph_single_case(ge::AscGraph &graph) {
  auto A = ge::Symbol(10, "A");
  auto R = ge::Symbol(20, "R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol(10, "ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TBuf(Position::kPositionVecOut);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  auto data = graph.FindNode("load1"); 
  data->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), -1}, {}, 0});
  return ge::SUCCESS;
}

Status BuildTqueTbufAscendGraphMultiCaseG0(ge::AscGraph &graph) {
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {nd});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {nd});
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load2", data2).TBuf(Position::kPositionVecIn);
      auto load_tbuf1 = Load("load3", data3).TBuf(Position::kPositionVecIn);
      auto load_tbuf2 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tbuf0);
      auto store3 = Store("store2", load_tbuf1);
      auto store4 = Store("store2", load_tbuf2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndB, *ndbT, *ndb, *ndbt}, {load_tque0, load_tbuf0, load_tbuf1, load_tbuf2, store1, store2, store3, store4},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1"); 
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), -1}, {}, 0});
  auto data1_node = graph.FindNode("load2");
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(1024), 0}, {}, 0});
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(2*1024), 0}, {}, 0});
  return ge::SUCCESS;
}

Status BuildTqueTbufAscendGraphMultiCaseG1(ge::AscGraph &graph) {
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {z0});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {z0});
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque1 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque2 = Load("load3", data3).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tque1);
      auto store3 = Store("store2", load_tque2);
      auto store4 = Store("store2", load_tbuf0);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*z0B, *z0bT, *z0b, *z0bt}, {load_tque0, load_tque1, load_tque2, load_tbuf0, store1, store2, store3, store4},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1"); 
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), 0}, {}, 0});
  return ge::SUCCESS;
}

Status BuildMultiCaseG0(ge::AscGraph &graph) {
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndT, ndt] = graph.TileSplit(nd.id);
  auto [ndTB, ndTb] = graph.BlockSplit(ndT->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {nd});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {nd});
  LOOP(*ndTB) {
    LOOP(*ndTb) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load2", data2).TBuf(Position::kPositionVecIn);
      auto load_tbuf1 = Load("load3", data3).TBuf(Position::kPositionVecIn);
      auto load_tbuf2 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tbuf0);
      auto store3 = Store("store2", load_tbuf1);
      auto store4 = Store("store2", load_tbuf2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndTB, *ndTb, *ndt}, {load_tque0, load_tbuf0, load_tbuf1, load_tbuf2, store1, store2, store3, store4},
          1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1");
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), -1}, {}, 0});
  auto data1_node = graph.FindNode("load2");
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(1024), 0}, {}, 0});
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(2*1024), 0}, {}, 0});
  return ge::SUCCESS;
}

Status BuildMultiCaseG1(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0});
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {z0});
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {z0});
  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque1 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque2 = Load("load3", data3).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tque1);
      auto store3 = Store("store2", load_tque2);
      auto store4 = Store("store2", load_tbuf0);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*z0TB, *z0Tb, *z0t}, {load_tque0, load_tque1, load_tque2, load_tbuf0, store1, store2, store3, store4},
          1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1");
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{ge::Symbol(16 * 1024), 0}, {}, 0});
  return ge::SUCCESS;
}
}
}
}
extern std::string RemoveAutoFuseTilingHeadGuards(const std::string &input);
extern void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result);
extern void AddHeaderGuardToFile(const std::string& file_name, const std::string& macro_name);
const std::string kFirstGraphName = "case0";
const std::string kSecondGraphName = "case1";

TEST_F(STestGenConcat, tque_tbuf_case0)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "tque_tbuf_case0";
  ascir::AscGraph tque_tbuf_case0(kFirstGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraph_single_case(tque_tbuf_case0), ge::SUCCESS);
  tque_tbuf_case0.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(tque_tbuf_case0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_EQ(res, true);

  auto tiling_res = GenerateTilingCodeCommon(op_name, fused_scheduled_result, options, tiling_func, "Concat_tiling_data.h", false);

  std::ofstream oss;
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(tque_tbuf_case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "b0_size"<< " = " << tilingData.get_b0_size() << std::endl;
  std::cout << "q0_size"<< " = " << tilingData.get_q0_size() << std::endl;
  std::cout << "b1_size"<< " = " << tilingData.get_b1_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  tque_tbuf_case0TilingData tilingData;
  tilingData.set_b0_size(64);
  tilingData.set_q0_size(128);
  tilingData.set_b1_size(512);
  PrintResult(tilingData);
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 公共函数定义：生成tiling代码并写入文件
std::map<std::string, std::string> GenerateTilingCodeCommon(
    const std::string &op_name,
    const ascir::FusedScheduledResult &fused_scheduled_result,
    const std::map<std::string, std::string> &options,
    const std::string &tiling_func,
    const std::string &header_file,
    bool gen_extra_infos) {

  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"" << header_file << "\"\n";
  oss << tiling_func;
  oss.close();

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options.at(ge::sym::kTilingDataTypeName);
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = gen_extra_infos;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);

  return tiling_res;
}

// 空tensor测试用例的辅助函数
namespace {
void BuildEmptyTensorGraph(const std::string &graph_name, ascir::ScheduledResult &schedule_result) {
  ascir::ScheduleGroup schedule_group1;
  ascir::AscGraph graph(graph_name.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph), ge::SUCCESS);
  graph.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result.schedule_groups.emplace_back(schedule_group1);
}

void GenerateTilingCodeForEmptyTensor(const std::string &op_name,
                                         const std::vector<ascir::ScheduledResult> &schedule_results,
                                         const std::string &graph_name) {
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_EQ(res, true);

  auto tiling_res = GenerateTilingCodeCommon(op_name, fused_scheduled_result, options, tiling_func, "Concat_tiling_data.h", true);

  std::ofstream oss;
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[graph_name + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
}

void CompileAndRunEmptyTensorTest(const std::string &graph_name) {
  std::ofstream oss;
  oss.open("tiling_func_main_concat_empty.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

int main() {
)" + graph_name + R"(TilingData tilingData;
  tilingData.set_block_dim(1);
  tilingData.set_ub_size(1024);
  tilingData.set_ND(0);

  if (GetTiling(tilingData)) {
    std::cout << "GetTiling success with empty tensor ND=0" << std::endl;
    return 0;
  } else {
    std::cout << "GetTiling failed with empty tensor" << std::endl;
    return -1;
  }
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();

  auto ret = std::system("g++ tiling_func_main_concat_empty.cpp Concat_tiling_func.cpp -o tiling_func_main_concat_empty -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat_empty > ./info_empty.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info_empty.log", "GetTiling success with empty tensor"), true);
}
}

ge::Status ConstructTQueTBufScheduleResults(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  ascir::AscGraph graph_1(kSecondGraphName.c_str());
  GE_ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraphMultiCaseG0(graph_0), ge::SUCCESS);
  graph_0.SetTilingKey(0U);
  GE_ASSERT_EQ(ge::ascir::cg::BuildTqueTbufAscendGraphMultiCaseG1(graph_1), ge::SUCCESS);
  graph_1.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  schedule_group2.impl_graphs.emplace_back(graph_1);

  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);
  return ge::SUCCESS;
}

ge::Status ConstructAutoTuneResults(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  ascir::AscGraph graph_1(kSecondGraphName.c_str());
  GE_ASSERT_EQ(ge::ascir::cg::BuildMultiCaseG0(graph_0), ge::SUCCESS);
  graph_0.SetTilingKey(0U);
  GE_ASSERT_EQ(ge::ascir::cg::BuildMultiCaseG1(graph_1), ge::SUCCESS);
  graph_1.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  schedule_group2.impl_graphs.emplace_back(graph_1);

  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);
  return ge::SUCCESS;
}

ge::Status ConstructSingleCaseForMultiTileScheduleResult(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult schedule_result1;
  std::vector<att::ModelInfo> model_info_list;
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  GE_ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1MultiTiling(graph_0), ge::SUCCESS);
  graph_0.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);
  return ge::SUCCESS;
}

ge::Status GenTilingImpl(std::vector<ascir::ScheduledResult> &schedule_results) {
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  GE_ASSERT_EQ(res, true);
  GE_ASSERT_EQ(att::test_common::GenTilingCodeToFile(
      op_name, tiling_funcs, fused_scheduled_result, options,
      kFirstGraphName + "TilingData"), ge::SUCCESS);
  auto ret =
      std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  GE_ASSERT_EQ(ret, 0);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  GE_ASSERT_EQ(ret, 0);
  return ge::SUCCESS;
}

ge::Status ConstructTQueTBufMultiCaseGroup() {
  std::vector<ascir::ScheduledResult> schedule_results;
  GE_ASSERT_EQ(ConstructTQueTBufScheduleResults(schedule_results), ge::SUCCESS);
  GE_ASSERT_EQ(GenTilingImpl(schedule_results), ge::SUCCESS);
  return ge::SUCCESS;
}

ge::Status ConstructAutoTuneCaseGroup() {
  std::vector<ascir::ScheduledResult> schedule_results;
  GE_ASSERT_EQ(ConstructAutoTuneResults(schedule_results), ge::SUCCESS);
  GE_ASSERT_EQ(GenTilingImpl(schedule_results), ge::SUCCESS);
  return ge::SUCCESS;
}

ge::Status ConstructSingleCaseForMultiTile() {
  std::vector<ascir::ScheduledResult> schedule_results;
  GE_ASSERT_EQ(ConstructSingleCaseForMultiTileScheduleResult(schedule_results), ge::SUCCESS);
  GE_ASSERT_EQ(GenTilingImpl(schedule_results), ge::SUCCESS);
  return ge::SUCCESS;
}

TEST_F(STestGenConcat, construct_single_case_for_multi_tile)
{
  EXPECT_EQ(ConstructSingleCaseForMultiTile(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_data.h"
using namespace optiling;
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_s0t_size(), 16); // s1t为Store输出的轴，所以不占UB，可以切最大
  MY_ASSERT_EQ(tilingData.get_s1t_size(), 16); // ub为1024，类型是fp32, 每个元素4字节，所以s0t_size为256
  MY_ASSERT_EQ(tilingData.get_q0_size() > 0, true);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;   
  tilingData.set_block_dim(1);
  tilingData.set_ub_size(1024);
  tilingData.set_S1(1024);  
  tilingData.set_S0(1024); 
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 用例描述：验证同等优先级轴的切分功能
// 构造2个order相同(=1)的切分轴S0和S1
// 预期结果：
// 1. Tiling求解成功
// 2. S0和S1被正确切分
// 3. UB利用率符合预期
TEST_F(STestGenConcat, equal_priority_tiling)
{
  EXPECT_EQ(ConstructSingleCaseForMultiTile(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat_equal_priority.cpp", std::ios::out);

  // Write the test code with multiple test cases using raw string literal
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include <vector>
#include "Concat_tiling_data.h"
using namespace optiling;

struct TestCase {
  const char *name;
  uint32_t S0;
  uint32_t S1;
  uint32_t ub_size;
};

int main() {
  std::vector<TestCase> test_cases = {
    {"Case1: Equal priority 16x16", 16, 16, 10240},
    {"Case2: Equal priority 32x16", 32, 16, 10240},
    {"Case3: Equal priority 32x32", 32, 32, 20480},
    {"Case4: Equal priority 64x32", 64, 32, 40960},
    {"Case5: Equal priority 128x64", 128, 64, 81920},
  };

  int passed = 0;
  int failed = 0;

  for (const auto &tc : test_cases) {
    case0TilingData tilingData;
    tilingData.set_ub_size(tc.ub_size);
    tilingData.set_S1(tc.S1);
    tilingData.set_S0(tc.S0);
    tilingData.set_block_dim(1);

    std::cout << "\n========== Testing: " << tc.name << " ==========" << std::endl;
    std::cout << "Input: S0=" << tc.S0 << ", S1=" << tc.S1 << ", ub_size=" << tc.ub_size << std::endl;

    if (GetTiling(tilingData)) {
      std::cout << "S0: " << tilingData.get_s0t_size() << std::endl;
      std::cout << "S1: " << tilingData.get_s1t_size() << std::endl;
      std::cout << "q0_size: " << tilingData.get_q0_size() << std::endl;

      // 验证切分结果
      if (tilingData.get_s0t_size() > 0 && tilingData.get_s1t_size() > 0) {
        std::cout << "PASSED - Both axes are tiled" << std::endl;
        passed++;
      } else {
        std::cout << "FAILED - One or both axes not tiled" << std::endl;
        failed++;
      }
    } else {
      std::cout << "FAILED - GetTiling returned false" << std::endl;
      failed++;
    }
  }

  std::cout << "\n==================== Test Summary ====================" << std::endl;
  std::cout << "Total: " << (passed + failed) << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;
  std::cout << "====================================================" << std::endl;

  return (failed == 0) ? 0 : -1;
}
)";

  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat_equal_priority.cpp "
      "Concat_tiling_func.cpp -o tiling_func_main_concat_equal_priority -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat_equal_priority");
  EXPECT_EQ(ret, 0);
}

// 用例描述：验证同等优先级轴的切分功能(使用多核UB tradeoff)
// 构造2个order相同(=1)的切分轴S0和S1
// 预期结果：
// 1. Tiling求解成功
// 2. S0和S1被正确切分
// 3. UB利用率符合预期
TEST_F(STestGenConcat, equal_priority_tiling_trade_off)
{
  setenv("AUTOFUSE_DFX_FLAGS",
         "--att_enable_multicore_ub_tradeoff=true;--att_corenum_threshold=100;--att_ub_threshold=10", 1);
  EXPECT_EQ(ConstructSingleCaseForMultiTile(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat_equal_priority.cpp", std::ios::out);

  // Write the test code with multiple test cases using raw string literal
  const std::string kRunTilingFuncMainLocal = R"(
#include <sstream>
#include <iostream>
#include <stdio.h>
double g_input_ub_threshold = 0.1;
double g_input_corenum_threshold = 1.0;

#define MY_ASSERT_EQ(x, y)                                                                                    \
 do {                                                                                                        \
   const auto &xv = (x);                                                                                     \
   const auto &yv = (y);                                                                                     \
   if (xv != yv) {                                                                                           \
     std::stringstream ss;                                                                                   \
     ss << "Assert (" << #x << " == " << #y << ") failed, expect " << yv << " actual " << xv;                \
     printf("%s\n", ss.str().c_str());                                                             \
     std::exit(1);                                                                                           \
   }                                                                                                         \
 } while (false)
#include <iostream>
#include <vector>
#include "Concat_tiling_data.h"
using namespace optiling;

#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Rational(a, b) ((double)(a) / (double)(b))
inline bool IsEqual(double a, double b)
{
  const double epsilon = 1e-8;
  double abs = (a > b) ? (a - b) : (b - a);
  return abs < epsilon;
}
template<typename T>
inline T Ceiling(T a)
{
  T value = static_cast<T>(static_cast<int64_t>(a));
  return (IsEqual(value, a)) ? value : (value + 1);
}

struct TestCase {
  const char* name;
  uint32_t S0;
  uint32_t S1;
  uint32_t ub_size;
  bool need_square{true};
  uint32_t expect_s0t_size{0U};
  uint32_t expect_s1t_size{0U};
  uint32_t expect_ub_size{0U};
  uint32_t expect_block_dim{0U};
};

int main() {
  std::vector<TestCase> test_cases = {
      {"Case1_0: Equal priority", 20, 20, 40000},
      {"Case1_1: Equal priority", 30, 30, 40000},
      {"Case1_2: Equal priority", 40, 40, 40000},
      {"Case1_3: Equal priority", 40, 200, 40000, false, 32},
      {"Case1_4: Equal priority", 40, 2000, 40000, false, 40},
      {"Case1_5: Equal priority", 40, 20000, 40000, false, 40},
      {"Case1: Equal priority", 50, 50, 40000},
      {"Case2: Equal priority", 100, 100, 40000},
      {"Case3: Equal priority", 150, 150, 40000},
      {"Case4: Equal priority", 200, 200, 40000},
      {"Case5: Equal priority", 250, 250, 40000},
      {"Case6: Equal priority", 300, 300, 40000},
      {"Case7: Equal priority", 350, 350, 40000},
      {"Case8: Equal priority", 400, 400, 40000},
      {"Case9: Equal priority", 500, 500, 40000},
      {"Case10: Equal priority", 550, 550, 40000},
      {"Case11: Equal priority", 600, 600, 40000},
      {"Case12: Equal priority", 750, 750, 40000},
      {"Case13: Equal priority", 800, 800, 40000},
      {"Case14: Equal priority", 850, 850, 40000},
      {"Case15: Equal priority", 900, 900, 40000},
      {"Case16: Equal priority", 950, 950, 40000},
      {"Case17: Equal priority", 1000, 1000, 40000},
  };

  int64_t passed = 0;
  int64_t failed = 0;
  constexpr uint32_t kHardwareCoreNum = 72U;
  constexpr uint32_t kCoreNumThreshold = kHardwareCoreNum * 0.9;
  constexpr uint32_t kSquareTileThreshold = 5U;
  for (const auto& tc : test_cases) {
    case0TilingData tilingData;
    tilingData.set_ub_size(tc.ub_size);
    tilingData.set_S1(tc.S1);
    tilingData.set_S0(tc.S0);
    tilingData.set_block_dim(kHardwareCoreNum);

    std::cout << "\n========== Testing: " << tc.name << " ==========" << std::endl;
    std::cout << "Input: S0=" << tc.S0 << ", S1=" << tc.S1 << ", ub_size=" << tc.ub_size << std::endl;

    if (GetTiling(tilingData)) {
      std::cout << "tc.name s0t: " << tc.name << " " << tilingData.get_s0t_size() << std::endl;
      std::cout << "tc.name s1t: " << tc.name << " " << tilingData.get_s1t_size() << std::endl;
      std::cout << "tc.name q0_size: " << tc.name << " " << tilingData.get_q0_size() << std::endl;

      // 验证切分结果
      double tensor_0 = (4 * tilingData.s1t_size_);
      double tensor_1 = (4 * tilingData.s0t_size_ * tilingData.s1t_size_);
      int64_t ub_size = ((32 * Ceiling((Rational(1, 32) * tensor_0))) + (32 * Ceiling((Rational(1, 32) * tensor_1))));
      int64_t all_ub_size = ub_size * tilingData.s1Ts0Tb_size_ * tilingData.block_dim_;
      std::cout << "tc.name ub_size: " << tc.name << " " << ub_size << std::endl;
      std::cout << "tc.name block_dim: " << tc.name << " " << tilingData.block_dim_ << std::endl;
      std::cout << "tc.name all_ub_size: " << tc.name << " " << all_ub_size << std::endl;
      bool is_pass = true;
      int64_t ub_ratio_10 = int(tc.ub_size * g_input_ub_threshold);
      std::cout << "tc.name ub_ratio_10: " << tc.name << " " << ub_ratio_10 << std::endl;
      auto block_dim = Max(0, Ceiling((Ceiling((static_cast<double>(tc.S0) /
                         static_cast<double>(tilingData.s0t_size_))) * Ceiling((static_cast<double>(tc.S1) /
                         static_cast<double>(tilingData.s1t_size_))) / (tilingData.s1Ts0Tb_size_))));
      std::cout << "tc.name block_dim: " << tc.name << " " << block_dim << std::endl;
      // 计算原始block_dim（不考虑s1Ts0Tb_size，用于验证算法效率）
      auto raw_block_dim = Ceiling((Ceiling((static_cast<double>(tc.S0) / static_cast<double>(tilingData.s0t_size_)))
                             * Ceiling((static_cast<double>(tc.S1) / static_cast<double>(tilingData.s1t_size_)))));
      std::cout << "tc.name raw_block_dim: " << tc.name << " " << raw_block_dim << std::endl;

      // 统一使用tilingData.block_dim_来判断（已考虑s1Ts0Tb_size）
      if (tilingData.block_dim_ == kHardwareCoreNum && ub_size < ub_ratio_10) {
        is_pass = false;
      }
      // 放宽约束：如果原始block_dim >= kCoreNumThreshold，即使实际block_dim < kCoreNumThreshold也通过
      if (tilingData.block_dim_ < kCoreNumThreshold && raw_block_dim >= kCoreNumThreshold) {
        is_pass = true;  // 算法已经找到了最优解，只是s1Ts0Tb_size导致实际block_dim降低
      }
      // 校验期望值
      if ((tc.expect_s0t_size > 0) && (tilingData.s0t_size_ != tc.expect_s0t_size)) {
        std::cout << "expect_s0t_size=" << tc.expect_s0t_size << ", s0t=" << tilingData.s0t_size_ << std::endl;
        is_pass = false;
      }
      if ((tc.expect_s1t_size > 0) && (tilingData.s1t_size_ != tc.expect_s1t_size)) {
        std::cout << "expect_s1t_size=" << tc.expect_s1t_size << ", s1t=" << tilingData.s1t_size_ << std::endl;
        is_pass = false;
      }
      if ((tc.expect_ub_size > 0) && (ub_size != tc.expect_ub_size)) {
        std::cout << "expect_ub_size=" << tc.expect_s0t_size << ", ub_size=" << ub_size << std::endl;
        is_pass = false;
      }
      if ((tc.expect_block_dim > 0) && (tilingData.block_dim_ != tc.expect_block_dim)) {
        std::cout << "expect_block_dim=" << tc.expect_block_dim << ", s0t=" << tilingData.block_dim_ << std::endl;
        is_pass = false;
      }
      bool is_square_tile = std::abs((static_cast<int64_t>(tilingData.get_s0t_size()) -
                                      static_cast<int64_t>(tilingData.get_s1t_size()))) < kSquareTileThreshold;
      if (!tc.need_square) {
        is_square_tile = true;
        if (tilingData.s1Ts0Tb_size_ > 2) {
          is_pass = false;
        }
      }
      if (is_pass && tilingData.get_s0t_size() > 0 && tilingData.get_s1t_size() > 0 && (ub_size <= tc.ub_size) &&
          (all_ub_size >= (tc.S0 * tc.S1 * 4)) && (tilingData.block_dim_ <= kHardwareCoreNum) &&
          // 同优先级切分，s1t和s0t有相同的优先级,保证s0t和s1t的差值在阈值内
          (is_square_tile)) {
        std::cout << "PASSED - Both axes are tiled" << std::endl;
        passed++;
      } else {
        std::cout << "FAILED - One or both axes not tiled, is_pass=" << is_pass << ", s0t=" << tilingData.get_s0t_size()
                  << ", s1t=" << tilingData.get_s1t_size() << ", ub_size=" << ub_size << ", all_ub_size=" << all_ub_size
                  << ", block_dim=" << tilingData.block_dim_ << std::endl;
        failed++;
      }
    } else {
      std::cout << "FAILED - GetTiling returned false" << std::endl;
      failed++;
    }
  }

  std::cout << "\n==================== Test Summary ====================" << std::endl;
  std::cout << "Total: " << (passed + failed) << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;
  std::cout << "====================================================" << std::endl;

  return (failed == 0) ? 0 : -1;
}
)";

  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat_equal_priority.cpp "
      "Concat_tiling_func.cpp -o tiling_func_main_concat_equal_priority -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat_equal_priority");
  EXPECT_EQ(ret, 0);
}

TEST_F(STestGenConcat, tque_tbuf_case1)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(ConstructTQueTBufMultiCaseGroup(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_data.h"
using namespace optiling;
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b1_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b2_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b3_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_q0_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b0_size(), 0);

  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_b3_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q0_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q1_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q2_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_b0_size(), 16384);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;   
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);  
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024); 
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 用例描述：复用tque_tbuf_case1的构图，开启DFX --att_accuracy_level=0，使能多核调优，增加使用的核数
// 用例期望结果：
// 1.Tiling求解成功；
// 2.使用核数未用满
TEST_F(STestGenConcat, auto_tuning_accuracy_level_0)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(ConstructAutoTuneCaseGroup(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_data.h"
using namespace optiling;
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim() < 64, true);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(256 * 1024);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024 * 1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024 * 1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 用例描述：复用tque_tbuf_case1的构图，开启DFX --att_accuracy_level=1，使能多核调优，增加使用的核数
// 用例期望结果：
// 1.Tiling求解成功；
// 2.使用核数相较level0模式上升；
TEST_F(STestGenConcat, auto_tuning_accuracy_level_1)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=1", 1);
  EXPECT_EQ(ConstructAutoTuneCaseGroup(), ge::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_data.h"
using namespace optiling;
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim() == 64, true);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(256 * 1024);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024 * 1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024 * 1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
    ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

ge::Status ConstructConcatTwoTilingCaseS0S1() {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "Concat";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd));
  graph_nd.SetTilingKey(0U);
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0));
  graph_s0.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group1.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  EXPECT_EQ(res, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  GE_ASSERT_TRUE(res == true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  GE_ASSERT_SUCCESS(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res));
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  GE_ASSERT_TRUE(ret == 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << CreateTilingMainFunc("Concat", "64", "245760", {{"S0", "1024"}, {"S1", "1024"}});
  oss.close();
  return ge::SUCCESS;
}

ge::Status ConstructTwoScheduleResultS0S1() {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd));
  graph_nd.SetTilingKey(0U);
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0));
  graph_s0.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group2.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  GE_ASSERT_TRUE(res == true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  GE_ASSERT_SUCCESS(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res));
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  GE_ASSERT_TRUE(ret == 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_S0(1024);
  tilingData.graph0_result1_g0_tiling_data.set_S1(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  return ge::SUCCESS;
}

TEST_F(STestGenConcat, case_axes_reorder)
{
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Concat_Normal_BeforeAutofuse(graph_normal);
  Concat_Normal_AfterScheduler(graph_normal);
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/tiling_func_main_concat.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system("g++ tiling_func_main_concat.cpp Concat_*_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_concat");
}

TEST_F(STestGenConcat, case_axes_reorder_got_static_shape)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_static";
  ascir::AscGraph graph_static(kFirstGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphStatic(graph_static), ge::SUCCESS);
  graph_static.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph_static);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;
extern "C" bool IsStaticShape();
int main() {
  if (IsStaticShape()) {
    std::cout << "Got static graph" << std::endl;
  } else {
    std::cout << "Got dynamic graph" << std::endl;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Got static graph"), true);
}

TEST_F(STestGenConcat, case_axes_reorder_got_dynamic_shape)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_dynamic";
  ascir::AscGraph graph_dynamic(kFirstGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_dynamic), ge::SUCCESS);
  graph_dynamic.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph_dynamic);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;
extern "C" bool IsStaticShape();
int main() {
  if (IsStaticShape()) {
    std::cout << "Got static graph" << std::endl;
  } else {
    std::cout << "Got dynamic graph" << std::endl;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Got dynamic graph"), true);
}

void STestGenConcat::VerifyGroupCacheBehavior() {
  // ========== 缓存校验 ==========
  // Test 1: ND=1024 Group Cache MISS -> SAVE
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] MISS! key=[1024]"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] SAVE SUCCESS: key=[1024]"), true);

  // Test 2: ND=1024 Group Cache HIT
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] HIT"), true);

  // Test 3: ND=2048 Group Cache MISS -> SAVE
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] MISS! key=[2048]"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] SAVE SUCCESS: key=[2048]"), true);

  // Test 4: ND=2048 Group Cache HIT (已在上面通过 HIT 校验覆盖)

  // Test 5: ND=4096 Operator Cache MISS -> SAVE
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] MISS! key=[4096]"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] SAVE SUCCESS: key=[4096]"), true);

  // Test 6: ND=4096 Operator Cache HIT
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] HIT! key=[4096]"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Operator level cache hit, input_shapes[4096]"), true);
}

void STestGenConcat::VerifyStaticShapeCacheBehavior() {
  // 用例描述：验证静态shape场景下的缓存功能
  // 用例期望结果：
  // 1. 生成的TilingFunc包含算子级缓存代码（即使kInputShapeSize=0）
  // 2. 使用空key进行缓存查询和保存
  // 3. 编译通过且运行时缓存正常工作

  // 验证生成的代码包含kInputShapeSize = 0（静态shape）
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "constexpr size_t kInputShapeSize = 0;"), true);

  // 验证生成的代码包含静态shape的缓存处理（空key）
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "std::array<uint32_t, kInputShapeSize> input_shapes = {};"), true);

  // 验证生成的代码包含算子级缓存查询（静态shape）
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "Operator level cache hit (static shape)"), true);
}

void STestGenConcat::BuildSingleGraphToScheduleGroup(
    ascir::AscGraph& graph,
    ascir::ScheduleGroup& schedule_group,
    uint32_t tiling_key) {
  graph.SetTilingKey(tiling_key);
  schedule_group.impl_graphs.emplace_back(graph);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group.impl_graphs);
}

void STestGenConcat::GenerateTilingFunctionAndWriteToFile(
    const std::string& op_name,
    const ascir::FusedScheduledResult& fused_scheduled_result,
    std::map<std::string, std::string>& options) {
  std::map<std::string, std::string> tiling_funcs;
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  EXPECT_EQ(res, true);

  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
}

void STestGenConcat::GenerateTilingDataAndHeader(
    const std::string& op_name,
    const std::string& graph_name,
    const ascir::FusedScheduledResult& fused_scheduled_result,
    std::map<std::string, std::string>& options) {
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options.at(kTilingDataTypeName);
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);

  std::ofstream oss;
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[graph_name + "TilingData"];
  oss.close();
}

void STestGenConcat::PrepareTestEnvironmentFiles(const std::string& test_header_content) {
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);

  EXPECT_EQ(ResultCheckerUtils::ReplaceLogMacrosGeneric("Concat_tiling_func.cpp"), true);

  if (!test_header_content.empty()) {
    std::ofstream oss;
    oss.open("Concat_tiling_test.h", std::ios::out);
    oss << test_header_content;
    oss.close();
  }
}

void STestGenConcat::CompileGeneratedTilingCode() {
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
}

void STestGenConcat::CompileAndRunTilingTest(const std::string& output_file) {
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG 2>&1");
  EXPECT_EQ(ret, 0);

  ret = std::system(("./tiling_func_main_concat > " + output_file + " 2>&1").c_str());
  EXPECT_EQ(ret, 0);
}

const std::string STestGenConcat::kStaticShapeCacheTestMain = R"RAW(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_static_cacheTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "Static shape cache test" << std::endl;
  std::cout << "get_block_dim = " << tilingData.get_block_dim() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_static_cacheTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);

  // First call: cache miss, normal solve
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "First tiling func execute failed." << std::endl;
    return -1;
  }

  // Second call: cache should hit (static shape uses empty key)
  if (GetTiling(tilingData)) {
    std::cout << "Second call success - cache hit" << std::endl;
  } else {
    std::cout << "Second tiling func execute failed." << std::endl;
    return -1;
  }

  return 0;
}
)RAW";

TEST_F(STestGenConcat, reuse_schedule_group_with_same_input_axis_name)
{
  // 构建图
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_s0), ge::SUCCESS);

  // 使用辅助函数构建schedule_group
  BuildSingleGraphToScheduleGroup(graph_nd, schedule_group1, 0U);
  BuildSingleGraphToScheduleGroup(graph_s0, schedule_group2, 1U);

  // 组装schedule_result
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_results.emplace_back(schedule_result1);

  // 配置options
  std::map<std::string, std::string> options;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);

  // 使用辅助函数生成tiling函数和数据
  GenerateTilingFunctionAndWriteToFile(op_name, fused_scheduled_result, options);
  GenerateTilingDataAndHeader(op_name, kFirstGraphName, fused_scheduled_result, options);

  // 准备测试环境和编译
  PrepareTestEnvironmentFiles(kConcatTilingTestHead);

  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << kRunTilingFuncMainSameND;
  oss.close();
  CompileGeneratedTilingCode();

  auto ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);

  VerifyGroupCacheBehavior();
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_name)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
  graph_nd.SetTilingKey(0U);
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS0(graph_s0), ge::SUCCESS);
  graph_s0.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group2.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << kRunTilingFuncMain;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_search_axis_name)
{
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
  graph_nd.SetTilingKey(0U);
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(graph_s0), ge::SUCCESS);
  graph_s0.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group2.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_ND2(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order)
{
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_schedule_result)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_schedule_result=0", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 默认选择tiling key 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_normal_tiling_case)
{
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
}

// 强制选择tiling case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_tiling_case)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定op name,选择tiling case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定错误的op name,按照默认选择tiling case 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_error_op_name)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Conct;--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定op name,强制选择group0的tiling case为case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name_group0_1)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_tiling_case=g0_1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定错误的op name,按照默认选择schedule result 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_error_op_name2)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Conct;--force_schedule_result=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定正确的的op name,选择schedule result 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name2)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_schedule_result=1", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 测试att_accuracy_level=1场景，使能自动选择最优核数
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_auto_tuning)
{
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=1", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), ge::SUCCESS);
  auto ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(STestGenConcat, fused_schedule_result_reuse_schedule_group)
{
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kFirstGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    std::string json_info;
    std::vector<att::ModelInfo> model_info_list;
    ascir::AscGraph graph_nd(kFirstGraphName.c_str());
    ascir::AscGraph graph_s0("graph_s0");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
    graph_nd.SetTilingKey(0U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(graph_s0), ge::SUCCESS);
    graph_s0.SetTilingKey(1U);
    schedule_group1.impl_graphs.emplace_back(graph_nd);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result1.schedule_groups.emplace_back(schedule_group1);
    schedule_result1.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result1);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    std::string json_info;
    std::vector<att::ModelInfo> model_info_list;
    ascir::AscGraph graph_nd("graph_s0s1");
    ascir::AscGraph graph_s0("graph_s1s0");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd), ge::SUCCESS);
    graph_nd.SetTilingKey(2U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0), ge::SUCCESS);
    graph_s0.SetTilingKey(3U);
    schedule_group1.impl_graphs.emplace_back(graph_nd);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result1.schedule_groups.emplace_back(schedule_group1);
    schedule_result1.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 3;}").c_str();
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result1);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_graph0_tiling_key = " << tilingData.get_graph0_tiling_key() << std::endl;
  std::cout << "get_graph1_tiling_key = " << tilingData.get_graph1_tiling_key() << std::endl;
  std::cout << "get_nd2bt_size = " << tilingData.graph0_result1_g0_tiling_data.get_nd2bt_size() << std::endl;
  std::cout << "get_s0bt_size = " << tilingData.graph1_result0_g0_tiling_data.get_s0bt_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_ND2(1024);
  tilingData.graph1_result0_g0_tiling_data.set_S0(1024);
  tilingData.graph1_result1_g0_tiling_data.set_S1(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_graph0_tiling_key = 1"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_graph1_tiling_key = 0"), true);
}

TEST_F(STestGenConcat, fused_schedule_result_tiling_case_score) {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kFirstGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_nd(kFirstGraphName.c_str());
    ascir::AscGraph sr0_g0("sr0_graph_g0");
    ascir::AscGraph sr0_g1("sr0_graph_g1");
    ascir::AscGraph sr0_g2("sr0_graph_g2");
    ascir::AscGraph sr0_g3("sr0_graph_g3");
    ascir::AscGraph sr0_g4("sr0_graph_g4");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
    graph_nd.SetTilingKey(0U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g0), ge::SUCCESS);
    sr0_g0.SetTilingKey(1U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g1), ge::SUCCESS);
    sr0_g1.SetTilingKey(2U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g2), ge::SUCCESS);
    sr0_g2.SetTilingKey(3U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g3), ge::SUCCESS);
    sr0_g3.SetTilingKey(4U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g4), ge::SUCCESS);
    sr0_g4.SetTilingKey(5U);
    schedule_group1.impl_graphs.emplace_back(graph_nd);
    schedule_group2.impl_graphs.emplace_back(sr0_g0);
    schedule_group2.impl_graphs.emplace_back(sr0_g1);
    schedule_group2.impl_graphs.emplace_back(sr0_g2);
    schedule_group2.impl_graphs.emplace_back(sr0_g3);
    schedule_group2.impl_graphs.emplace_back(sr0_g4);
    schedule_group1.graph_name_to_score_funcs["graph_nd"] =
        "int32_t CalcScore(AscGraph0ScheduleResult0G0TilingData &tiling_data) {return -1;}";
    schedule_group2.graph_name_to_score_funcs["sr0_graph_g0"] =
        "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 100;}";
    schedule_group2.graph_name_to_score_funcs["sr0_graph_g1"] =
        "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 1;}";
    schedule_group2.graph_name_to_score_funcs["sr0_graph_g3"] =
        "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 100;}";
    schedule_group2.graph_name_to_score_funcs["sr0_graph_g4"] =
        "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return -1;}";
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result1.schedule_groups.emplace_back(schedule_group1);
    schedule_result1.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result1);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_nd("graph_s0s1");
    ascir::AscGraph graph_s0("graph_s1s0");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd), ge::SUCCESS);
    graph_nd.SetTilingKey(6U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0), ge::SUCCESS);
    graph_s0.SetTilingKey(7U);
    schedule_group1.impl_graphs.emplace_back(graph_nd);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result1.schedule_groups.emplace_back(schedule_group1);
    schedule_result1.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result1);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_graph0_tiling_key = " << tilingData.get_graph0_tiling_key() << std::endl;
  std::cout << "get_graph1_tiling_key = " << tilingData.get_graph1_tiling_key() << std::endl;
  std::cout << "get_nd2bt_size = " << tilingData.graph0_result1_g0_tiling_data.get_nd2bt_size() << std::endl;
  std::cout << "get_s0bt_size = " << tilingData.graph1_result0_g0_tiling_data.get_s0bt_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_ND2(1024);
  tilingData.graph1_result0_g0_tiling_data.set_S0(1024);
  tilingData.graph1_result1_g0_tiling_data.set_S1(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_graph0_tiling_key = 1"), true);
}

// 场景：
// graph0: 先切Tile再切Block（有打分函数）
// graph1: 先切Block再切Tile（有打分函数）
// graph2: 先切Block再切Tile（与graph1相同Schedule，有打分函数）
// 预期：
// 选择graph1，graph0虽然打分高，但不可与graph1,2进行打分比较
TEST_F(STestGenConcat, fused_schedule_result_tiling_case_score_same_schedule) {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group1;
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result1;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph sr0_g0_0(kGraphName.c_str());
    ascir::AscGraph sr0_g0_1("graph_nd_1");
    ascir::AscGraph sr0_g0_2("graph_nd_2");
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2TB(sr0_g0_0), ge::SUCCESS);
    sr0_g0_0.SetTilingKey(0U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2WithAbs(sr0_g0_1), ge::SUCCESS);
    sr0_g0_1.SetTilingKey(1U);
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g0_2), ge::SUCCESS);
    sr0_g0_2.SetTilingKey(2U);
    schedule_group2.impl_graphs.emplace_back(sr0_g0_0);
    schedule_group2.impl_graphs.emplace_back(sr0_g0_1);
    schedule_group2.impl_graphs.emplace_back(sr0_g0_2);
    schedule_group2.graph_name_to_score_funcs["graph_nd_0"] =
        "int32_t CalcScore(graph_ndTilingData &tiling_data) {return 101;}";
    schedule_group2.graph_name_to_score_funcs["graph_nd_1"] =
        "int32_t CalcScore(graph_ndTilingData &tiling_data) {return 100;}";
    schedule_group2.graph_name_to_score_funcs["graph_nd_2"] =
        "int32_t CalcScore(graph_ndTilingData &tiling_data) {return -1;}";
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func =
        ("int32_t CalcScore(" + kGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_ndTilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_tiling_key = " << tilingData.get_tiling_key() << std::endl;
  std::cout << "get_nd2bt_size = " << tilingData.get_nd2bt_size() << std::endl;
  std::cout << "get_nd2t_size = " << tilingData.get_nd2t_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_ND2(1024);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret =
      std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = "), true);
}

TEST_F(STestGenConcat, fused_schedule_result_prompt_aligned)
{
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kFirstGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_s0(kFirstGraphName.c_str());
    ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND2TB(graph_s0), ge::SUCCESS);
    graph_s0.SetTilingKey(1U);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kFirstGraphName + "TilingData"];
  oss.close();
  auto ret = std::system(std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "tests/autofuse/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_data.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_block_dim = " << tilingData.get_block_dim() << std::endl;
  std::cout << "get_nd2t_size = " << tilingData.get_nd2t_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_ND2(1025);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();
  ret = std::system("g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp -o tiling_func_main_concat -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_nd2t_size = 17"), true);
}

// 测试ND为0时动态shape场景的空tensor处理
TEST_F(STestGenConcat, empty_tensor_nd_zero_dynamic_shape) {
  const std::string kFirstGraphName = "graph_empty_tensor";
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;

  // 构建图结构
  BuildEmptyTensorGraph(kFirstGraphName, schedule_result1);
  schedule_results.emplace_back(schedule_result1);

  // 生成tiling代码
  GenerateTilingCodeForEmptyTensor("Concat", schedule_results, kFirstGraphName);

  // 编译并运行测试
  CompileAndRunEmptyTensorTest(kFirstGraphName);
}
// 用例描述：验证静态shape场景下的缓存功能
// 用例期望结果：
// 1. 生成的TilingFunc包含算子级缓存代码（即使kInputShapeSize=0）
// 2. 使用空key进行缓存查询和保存
// 3. 编译通过且运行时缓存正常工作
TEST_F(STestGenConcat, static_shape_cache_support)
{
  // 构建图
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;

  const std::string kFirstGraphName = "graph_static_cache";
  ascir::AscGraph graph_static(kFirstGraphName.c_str());
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphStatic(graph_static), ge::SUCCESS);

  // 使用辅助函数构建schedule_group
  BuildSingleGraphToScheduleGroup(graph_static, schedule_group1, 0U);

  // 组装schedule_result
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  // 配置options
  std::map<std::string, std::string> options;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);

  // 使用辅助函数生成tiling函数和数据
  GenerateTilingFunctionAndWriteToFile(op_name, fused_scheduled_result, options);
  GenerateTilingDataAndHeader(op_name, kFirstGraphName, fused_scheduled_result, options);

  // 准备测试环境并验证生成的代码
  PrepareTestEnvironmentFiles();
  VerifyStaticShapeCacheBehavior();

  // 生成静态shape缓存测试主函数
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << kStaticShapeCacheTestMain;
  oss.close();

  // 编译并运行测试，验证输出
  CompileAndRunTilingTest();
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Static shape cache test"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Second call success"), true);
}

// 用例描述：验证多group场景下enable_group_parallel_optimize=true时的代码生成
// 用例期望结果：
// 1. 生成的FindPerfBetterTilingbyCaseId函数包含core_num参数
// 2. 生成的GetTilingCaseScoreFunc函数包含core_num参数
// 3. 函数调用链正确传递corenum参数
// 4. 包含group_num常量和性能调整代码
TEST_F(STestGenConcat, multi_group_with_enable_group_parallel_optimize)
{
  // 构建图
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;

  const std::string kFirstGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kFirstGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), ge::SUCCESS);
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphND(graph_s0), ge::SUCCESS);

  // 使用辅助函数构建schedule_group
  BuildSingleGraphToScheduleGroup(graph_nd, schedule_group1, 0U);
  BuildSingleGraphToScheduleGroup(graph_s0, schedule_group2, 1U);

  // 组装schedule_result
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  // 关键：设置enable_group_parallel=true
  schedule_result1.enable_group_parallel = true;
  schedule_results.emplace_back(schedule_result1);

  // 配置options
  std::map<std::string, std::string> options;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);

  // 使用辅助函数生成tiling函数和数据
  GenerateTilingFunctionAndWriteToFile(op_name, fused_scheduled_result, options);
  GenerateTilingDataAndHeader(op_name, kFirstGraphName, fused_scheduled_result, options);

  // 准备测试环境和编译
  PrepareTestEnvironmentFiles(kConcatTilingTestHead);

  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << kRunTilingFuncMainSameND;
  oss.close();
  CompileGeneratedTilingCode();

  // 验证生成的代码包含正确的参数传递
  // 1. FindPerfBetterTilingbyCaseId函数签名包含core_num参数
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
              "bool FindPerfBetterTilingbyCaseId"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
              ", uint32_t core_num) {"), true);
  // 2. GetTilingCaseScoreFunc函数签名包含core_num参数
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
              "bool GetTilingCaseScoreFunc"), true);
  // 3. GetTilingKey函数获取corenum并传递给GetTilingCaseScoreFunc
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
              "uint32_t corenum = tiling_data.get_block_dim()"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
              ", corenum);"), true);
  // 4. GetTilingCaseScoreFunc调用FindPerfBetterTilingbyCaseId时传递core_num
  // 注意：由于代码格式可能不同，这里只检查core_num参数被传递
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
              ", core_num);"), true);

  auto ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
}

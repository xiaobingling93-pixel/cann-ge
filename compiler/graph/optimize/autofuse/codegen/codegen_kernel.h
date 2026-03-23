/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_KERNEL_H__
#define __CODEGEN_KERNEL_H__

#include <set>
#include <utility>
#include "ascir.h"
#include "ascgen_log.h"
#include "ascir_ops_utils.h"
#include "schedule_result.h"

namespace codegen {
class Code {
 public:
  virtual std::string Str() const = 0;
};

class Type : public Code {
 public:
  const std::string name;

  explicit Type(const std::string& type_name);
  std::string Str() const override;
};

const Type kVoidT{"void"};
const Type kIntT{"int"};
const Type kInt32T{"int32_t"};
const Type kInt64T{"int64_t"};
const Type kUint32T{"uint32_t"};
const Type kHalfT{"half"};
const Type kGmAddrT{"GM_ADDR"};

class Variable : public Code {
public:
  const Type type;
  const std::string name;

  Variable(const Type &var_type, const std::string &var_name);
  std::string Str() const override;
  std::string AsArg() const;
  std::string Define(std::string &&init = "", bool define_const = false) const;
  inline std::string DefineConst(std::string &&init = "") const {
    return Define(std::move(init), true);
  }
  std::string DefineConstReciprocal() const;
  std::string Assign(std::string &value) const;
};

struct Int : public Variable {
  explicit inline Int(const std::string &int_name) : Variable(kIntT, int_name) {}
};

struct Int64 : public Variable {
  explicit inline Int64(const std::string &int64_name) : Variable(kInt64T, int64_name) {}
};

struct GM_ADDR : public Variable {
  explicit inline GM_ADDR(std::string gm_addr_name) : Variable{kGmAddrT, gm_addr_name} {};
};

struct Uint32 : public Variable {
  explicit inline Uint32(const std::string &uint32_name) : Variable(kUint32T, uint32_name) {}
};

class Axis : public ascir::Axis, public Variable {
 public:
  using ascir::Axis::type;
  explicit Axis(const ascir::Axis &axis);

  Variable loop_size;
  Int elem_size;
  Int64 actual_size;
  Int64 axis_size;
  Int tail_size;

  ascir::SizeExpr elem_size_expr;
  ascir::SizeExpr actual_size_expr;
  ge::Expression size_expr;

  bool is_split_b { false };
};

class Tensor : public Variable {
 public:
  static Status DtypeName(ge::DataType dtype, std::string &dtype_name);
  static const Type GlobalTensorTypes(std::string &dtype_name);
  static const Type LocalTensorTypes(std::string &dtype_name);

  ascir::TensorId id;
  ascir::ReuseId reuse_id;
  ge::DataType dtype;
  ge::AllocType alloc_type;
  ascir::Position position;

  vector<ascir::AxisId> axis;
  vector<ascir::SizeExpr> axis_size;
  vector<ascir::SizeExpr> axis_strides;
  vector<uint32_t> vectorized_axis_pos;
  vector<ascir::AxisId> vectorized_axis;
  vector<ascir::SizeExpr> vectorized_strides;

  ascir::QueId que_id;
  ascir::BufId buf_id;

  Uint32 size; /** Que/Buf size in element number */
  Uint32 actual_size;
  Uint32 que_depth;
  Uint32 que_buf_num;
  Uint32 que_share_offset;
  std::string share_pre_size{"0"}; /* 共用场景，前一个size */

  std::string const_value; /** Constant node value */
  ascir::SizeExpr const_value_expr; /** Constant node value */
  uint32_t que_depth_value;
  uint32_t que_buf_num_value;

  ascir::MergeScopeId merge_scope;
  bool is_constant{false};
  bool is_ub_scalar{false};
  bool is_load_link_store_and_vec{false}; // 该tensor 是load 直连store, load连vec的场景
  bool need_gen_get_value_of_ub_scalar{false};
  bool need_duplicate_value_of_ub_scalar{false};

  std::string ub_scalar_name;
  bool isAr{false};
  mutable bool no_need_realloc = false;

  Status InitUbScalar(std::string &result) const;
  Status GenDuplicateValueOfUbScalar(std::string &result) const;
  Status DefineUbScalar(std::string &result) const;
  explicit Tensor(const ascir::TensorAttr& tensor, std::string &dtype_name, const std::string& tensor_name = "");
  explicit Tensor(const ascir::TensorAttr& tensor, std::string &dtype_name, const ascir::SizeExpr& value, const std::string& tensor_name = "");
  explicit Tensor(const std::string& value, const ascir::TensorAttr& tensor, std::string &dtype_name, const std::string& tensor_name = "");
  Status Init();

  // For GlobalTensor
  Status SetGlobalBuffer(GM_ADDR global, const std::string& offset, std::string &result) const;

  bool IsUbScalar() const {
    return is_ub_scalar && need_gen_get_value_of_ub_scalar;
  }
  bool IsConstScalar() const  {
    return is_constant;
  }
  bool IsAnyScalar() const  {
    return IsUbScalar() || IsConstScalar();
  }
  std::string GetScalarValue() const {
    return IsUbScalar() ? ub_scalar_name : const_value;
  }
  inline ascir::SizeExpr GetTensorSize() const {
    ascir::SizeExpr size_expr = ge::Symbol(1);
    for (size_t i = 0; i < axis_size.size(); i++) {
      size_expr = ge::sym::Mul(size_expr, axis_size[i]);
    }
    return size_expr;
  }
};
using TensorPtr = std::shared_ptr<Tensor>;
using ConstTensorPtr = std::shared_ptr<const Tensor>;
Status PositionValue(ascir::Position position, std::string &result);

class MergeScope {
 public:
  ascir::MergeScopeId id;
  ascir::Position position;
  std::vector<ascir::TensorId> tensors;

  Uint32 size;
  Uint32 depth;
  Uint32 buf_num;

  MergeScope(ascir::MergeScopeId merge_scope_id, ascir::Position pos);
};

class TQue : public Variable {
 public:
  ascir::QueId id;
  ascir::Position position;
  std::set<ascir::MergeScopeId> merge_scopes;
  std::set<ascir::TensorId> not_merge_tensors;
  std::map<ascir::ReuseId, std::vector<ascir::TensorId>> share_group;

  Uint32 size;
  Uint32 depth;
  Uint32 buf_num;

  Variable buf;
  bool is_cv_ub_fusion{false};

  TQue(ascir::QueId que_id, ascir::Position pos, std::string &position_name);
  TQue(ascir::QueId que_id,
       ascir::Position src_position,
       const std::string &src_position_name,
       const std::string &dst_position_name);
  std::string AllocBuf(const bool with_define = true) const;
  std::string EnqueBuf() const;
  std::string DequeBuf(const bool is_unit_first) const;
  std::string FreeBuf() const;
};

class TBuf : public Variable {
 public:
  ascir::BufId id;
  ascir::Position position;
  std::set<ascir::MergeScopeId> merge_scopes;
  std::set<ascir::TensorId> not_merge_tensors;
  std::vector<ge::Expression> tmp_buf_size_list;

  Uint32 size;

  Variable buf;

  bool tmp_buf_reuse{false};
  TBuf(ascir::BufId buf_id, const ascir::Position pos, std::string &position_name);
  std::string AllocBuf(const bool with_define = true) const;
  std::string AllocBuf(std::string buf_name, std::string dtype_name, const bool with_define = true) const;
};

class Tiler : public Code {
 public:
  Variable tiling_data;
  Variable gm_tiling;

  Int block_dim;
  std::map<ascir::AxisId, codegen::Axis> axis_map;
  std::vector<std::pair<ge::Expression, ge::Expression>> sizes;
  mutable std::vector<std::pair<ge::Expression, ge::Expression>> actual_sizes;

  explicit Tiler(const std::string &tiling_data_type = "TilingData", const std::string &tiling_data_name = "t");

  // parse
  void AddSizeVar(ascir::SizeVar size);
  Status AddAxis(const ascir::Axis &axis);
  bool IsFrom(ascir::AxisId src, ascir::AxisId dst) const;
  bool HasSameOriginAxis(ascir::AxisId src, ascir::AxisId dst) const;
  void AddAxisSplitBAttr();

  /* TilingCaseId读取 */
  uint32_t GetTilingCaseId() const;
  void SetTilingCaseId(uint32_t tilingCaseId);
  void EnableGroupParallel(bool enable_group_parallel);

  // generate
  std::string Str() const override;
  std::string Size(const ascir::SizeExpr& size, bool using_int_tiling_data=false) const;
  std::string ActualSize(const ascir::SizeExpr& size, bool using_int_tiling_data=false) const;
  std::string Offset(const std::vector<ascir::AxisId> &current_axis, const std::vector<ascir::AxisId> &axis,
                     const std::vector<ascir::SizeExpr> &strides) const;
  std::string TensorVectorizedOffset(const std::vector<ascir::AxisId> &current_axis, const Tensor &tensor) const;
  std::string TensorVectorizedSize(const Tensor &tensor) const;
  std::string ShareTensorVectorizedSize(const Tensor &tensor) const;
  std::string TensorActualSize(const Tensor &tensor) const;
  const Axis& GetAxis(const ascir::AxisId id) const;
  std::string AxisSize(const ascir::AxisId id) const;
  std::string AxisSize(const Axis& axis) const;
  std::string AxisName(const ascir::AxisId axis_id) const;

  /**
   * 定义Block外轴，并用blockidx初始化。
   */
  std::string BlockOutterAxisDefine();
  void BlockOutterAxisDefine(const ascir::AxisId id, std::stringstream &ss);
  /* 尾块生成处理 */
  std::string GenAxisSizeNew(const ascir::AxisId id) const;
  std::string GenInnerLoopSizeAndActualSize(const ascir::AxisId id, const ascir::AxisId loop_axis,
                                            bool is_need_divide_sum, bool is_define = true) const;
  std::string CalcFromAxis(const ascir::AxisId id, bool is_define = true) const;

 private:
  uint32_t tiling_case_id;
  bool enable_group_parallel_ = false;
};

class TPipe : public Variable {
 public:
  const Tiler& tiler;
  Variable tmp_buf;
  map<ascir::TensorId, Tensor> tensors;
  map<ascir::MergeScopeId, MergeScope> merge_scopes;
  map<ascir::QueId, TQue> ques;
  map<ascir::BufId, TBuf> bufs;
  std::set<ascir::QueId> load_store_qids;
  std::set<ascir::QueId> non_load_store_qids;
  std::string reuse_dtype_name = "";
  std::vector<ascir::TensorId> need_gen_blk_tensors;
  ascir::CubeTemplateType cv_fusion_type{ascir::CubeTemplateType::kDefault};
  ascir::TensorId cube_output_tensor_id = ge::kIdNone;
  ascir::TensorId cube_output_que_id = ge::kIdNone;
  std::vector<ascir::BufId> contiguous_buf_ids;

  TPipe(const std::string &tpipe_name, const Tiler &tpipe_tiler);
  Status AddTensor(const Tensor &tensor);
  Status AddTensor(const ascir::TensorAttr &tensor_attr, const std::string& tensor_name = "");
  Status AddTensor(const std::string& const_value, const ascir::TensorAttr &tensor_attr, const std::string& tensor_name = "");
  Status AddTensor(const ascir::TensorAttr &tensor_attr, const ascir::SizeExpr& const_value, const std::string& tensor_name = "");

  const TQue *GetQue(const ascir::QueId id) const;
  const TBuf &GetBuf(const ascir::BufId id) const;
  const Tensor* GetTensor(ascir::TensorId id) const;
  Tensor* GetTensor(ascir::TensorId id);

  Status InitTQueBuffers(const TQue &que, std::string &result) const;
  Status InitTBufBuffer(const TBuf &buf, std::string &result) const;

  Status TensorAlloc(const Tensor& tensor, std::string &result) const;
  std::string TensorSizeCalc() const;
  std::string TensorActualSizeCalc(const ascir::TensorId id) const;
  Status MergeScopeSizeCalc(std::string &result) const;
  Status LocalTBufAlloc(const TBuf &buf, std::string &result, const bool with_define = true) const;
  Status LocalTBufAllocLoopTwice(std::string &result, const bool with_define = true) const;
  std::string AllocTmpBuf(const TBuf &buf, const bool with_define = true) const;
  std::string GenDuplicateBufAlloc(const std::set<std::pair<std::string, std::string>>& pre_api_extract_dup) const;
  Status LocalTQueAlloc(std::string &result) const;
  Status LocalTensorQueBufAlloc(std::string &result) const;
  std::string SyncMte3ToMte2(const Tensor in_tensor) const;
  std::string SyncMte2ToMte3(const Tensor in_tensor) const;
  Status CollectQues(const ascir::ImplGraph &graph);
  void AddBlkTensor(const ascir::TensorId &tensorid) {
    need_gen_blk_tensors.emplace_back(tensorid);
  }
  bool IsNeedGenBlkTensor(const ascir::TensorId& tensorid) const {
    return std::find(need_gen_blk_tensors.begin(), need_gen_blk_tensors.end(), tensorid) != need_gen_blk_tensors.end();
  }
  Status BlkTensorAllocAndInit(std::string &result) const;
  void SetUsingAttCalcQBTSizeConfig(bool using_att_calc_qbt_size);
  Status GetCVFusionCubeOutputUBTensorIdAndQueId(const ascir::ImplGraph &graph);
  Status LocalTensorDefine(std::string &result) const;
  Status LocalTBufAssign(const TBuf &buf, std::string &result) const;
  std::string TensorSizeDefine() const;
  Status TensorSizeAssign(std::string dtype_name, std::string &result) const;
  std::string GenDuplicateBufDefine(const std::set<std::pair<std::string, std::string>>& pre_api_extract_dup) const;
  std::string GenDuplicateBufAssign(const std::set<std::pair<std::string, std::string>>& pre_api_extract_dup) const;
  Status BlkTensorDefine(std::string &result) const;
  Status BlkTensorAssign(std::string &result) const;
 private:
  Status ParseTBufReuse(TBuf buf, std::string& reuse_dtype_name, bool& is_buf_reuse,
                        std::vector<const Tensor *>& reuse_buf_tensors, std::stringstream &tensor_size_max) const;
  bool using_att_calc_qbt_size_ = true;
};

struct ApiCall;
struct Loop;
enum class LoopType : int8_t {
  CALL = 0,
  LOOP
};
enum class BoolType : int8_t {
  FALSE = 0,
  TRUE = 1,
  FAILED = 2
};

struct LoopBody {
  LoopType type;
  union {
    ApiCall *call;
    Loop *loop;
  };
};

struct ApiTensor {
  ascir::TensorId id;
  ascir::ReuseId reuse_id;
  struct ApiTensor* reuse_from;
  struct ApiTensor* reuse_next;
  struct ApiTensor* share_prev;
  struct ApiTensor* share_next;
  int32_t share_order;
  const ApiCall* write;
  std::vector<const ApiCall*> reads;

  ApiTensor();
};

// todo: 后面api_attr中属性都要收编到子类中, 收编完成之后, 删除api_attr; 不允许在ApiAttr中新增字段
struct ApiAttr {
  ge::Expression offset;
  float negative_slope = 0.0;
  int64_t gather_axis = 0;
  bool negative_index_support = false;
};

enum class ApiScene : int8_t {
  kDefault = 0,          // 非CV融合场景
  kCVFuseUBLoad,         // CV融合场景, load节点的输入tensor在UB上(Cube的输出)
};
 
enum class ComputeStage : int8_t {
  kDefault = 0,          // 非CV融合场景
  kCVFuseStage1,         // CV融合场景阶段1, Cube输出Tensor的生命周期之内
  kCVFuseStage2,         // CV融合场景阶段2, Cube输出Tensor的生命周期之外
};
 
struct ApiCallContext {
  ApiScene scene = ApiScene::kDefault;
  ComputeStage stage = ComputeStage::kDefault;
 
  bool isCVFusion() const {
    return scene != ApiScene::kDefault;
  }
};

class ApiCall {
 public:
  // Constructor and Destructor
  virtual ~ApiCall() = default;
  explicit ApiCall(const std::string &api_name) noexcept : api_name_(api_name) {}

  // Public Member Function
  virtual Status Init(const ascir::NodeView &node);
  virtual Status ParseAttr(const ascir::NodeView &node);
  virtual Status PreProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                            std::string &result) const;
  virtual Status GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const;
  virtual Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                          const std::vector<std::reference_wrapper<const Tensor>> &input,
                          const std::vector<std::reference_wrapper<const Tensor>> &output, std::string &result) const;
  virtual Status PostProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                             const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                             std::string &result) const;
  virtual Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                          std::string &result) const;
  virtual Status GenerateMacro(std::string &result) const;
  virtual bool AreContiguousBufsPreferred() const {
    return false;
  };

  bool FreeInputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool FreeUnusedOutputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool SyncOutputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool WaitInputs(const TPipe &tpipe, std::stringstream &ss) const;
  bool IsReadOutersideWrite(ascir::AxisId &target_id) const;
  Status AllocOutputs(const TPipe &tpipe, std::stringstream &ss, bool create_sync = true) const;
  bool IsUnitLastRead(const ApiTensor &tensor) const;

  // Public Member Variables
  std::string api_name_;
  ascir::AxisId axis;
  std::string type; // ascir tpye
  int64_t depth;
  ascir::ComputeUnit unit;
  ascir::ComputeType compute_type;
  std::vector<ApiTensor> outputs;
  std::vector<const ApiTensor *> inputs;
  bool enable_cache{false};
  bool is_input_tbuf_contiguous = false;
  std::string enable_cache_with_condition;
  // 用于标记Call节点执行状态
  // broadcast cache场景：在Call节点外生成控制条件
  ge::ExecuteCondition exec_condition;
  // todo: 后面api_attr中属性都要收编到子类中, 收编完成之后, 删除api_attr; 不允许在ApiAttr中新增字段
  ApiAttr api_attr;
  ApiCallContext api_call_context = {ApiScene::kDefault, ComputeStage::kDefault};
  std::unordered_map<int64_t, int64_t> tmp_buf_id;

 private:
  bool WaitInputVector(const TPipe &tpipe, const ApiTensor *in, const Tensor &t, std::stringstream &ss) const;
  bool WaitInputMte(const TPipe &tpipe, const ApiTensor *in, const Tensor &t, std::stringstream &ss) const;
  BoolType WaitShareInputs(const TPipe &tpipe, const ApiTensor *in, const Tensor t, std::stringstream &ss) const;
  BoolType AllocShareOutputs(const TPipe &tpipe, const ApiTensor &out, const Tensor t, std::stringstream &ss) const;
  Status HandleVecOutAlloc(const TPipe &tpipe, const ApiTensor &out, const Tensor &t, std::stringstream &ss,
                           bool with_define) const;

 private:
 // 新增字段

};

struct Loop {
  ascir::AxisId axis_id;
  struct Loop* parent;
  std::vector<LoopBody> bodys;
  std::set<const ApiCall *> used_calls = {};
  bool is_graph_has_reduce_node = false;  // 当前图上是否有reduce节点
  bool is_ar = false;                     // 如果图上有reduce节点  是否为AR
  explicit Loop(const ascir::AxisId axis);
  ComputeStage compute_stage = ComputeStage::kDefault;

  void AddLoop(Loop *loop);
  void AddCall(ApiCall *call);

  /* 将会通过new 申请内存，需要通过Destruct释放 */
  Status ConstructFromNodes(ascir::NodeViewVisitorConst nodes, const Tiler &tiler, TPipe& tpipe);
  void Destruct();

  Status Generate(const Tiler& tiler, const TPipe& tpipe, std::string &result,
                  ComputeStage stage = ComputeStage::kDefault);
  bool IsReduceAxisNeedDivideSum(const TPipe &tpipe) const;
  const Tensor& GetReduceApiTensor(const TPipe &tpipe, bool is_input = false) const;
  void CollectTensorCrossLoop(std::map<ascir::AxisId, std::vector<ApiCall *>> &api_calls);
  Status ActualSizeDefine(const Tiler &tiler, const TPipe &tpipe, std::string &result);

 private:
  Status GenerateLoop(const Tiler& tiler, const TPipe& tpipe, std::vector<ascir::AxisId>& current_axis, std::stringstream& ss);
  Status GenerateBody(const Tiler& tiler, const TPipe& tpipe, std::vector<ascir::AxisId>& current_axis,
                      std::stringstream& ss);
  void GenerateEnCacheCondition(const Tiler &tiler, const TPipe &tpipe, const Axis &axis, std::stringstream &ss) const;
  bool IsFindInUsedCalls(const ApiCall *call) const;
  std::string GetReduceType() const;
  bool IsBodyContainLoop() const;
};

struct ReduceOpType {
  static constexpr int32_t kMin = 0;
  static constexpr int32_t kMax = 1;
  static constexpr int32_t kSum = 2;
  static constexpr int32_t kProd = 3;
  static constexpr int32_t kAny = 4;
  static constexpr int32_t kAll = 5;
  static constexpr int32_t kMean = 6;
};

struct CodegenConfig {
  bool is_inductor;
  bool using_att_calc_qbt_size; // 是否使用att计算tque/tbuf/tmpbuf
};

class KernelUtils {
 public:
  static std::string Max();
  static std::string Sum();
  static Status BlkNum(ge::DataType dtype, std::string &result);
  static Status BlkAlign(ge::DataType dtype, std::string &result);
  static std::string SizeAlign();
  static std::string FindNearestPower2();
};

struct TilingFuncCall {
  TilingFuncCall(std::string func_call, bool has_workspace_node, bool need_sync_all) :
        func_call_(std::move(func_call)), has_workspace_node_(has_workspace_node), need_sync_all_(need_sync_all) {}
  std::string func_call_;
  bool has_workspace_node_;
  bool need_sync_all_;
};

class Kernel {
 public:
  GM_ADDR workspace_arg;
  std::vector<GM_ADDR> inputs;
  std::vector<ascir::TensorId> input_tensors;
  std::vector<GM_ADDR> outputs;
  std::vector<ascir::TensorId> output_tensors;
  std::vector<ascir::TensorId> constant_tensors;
  std::vector<ascir::TensorId> ub_scalar_tensors;
  std::vector<Uint32> workspaces;
  std::map<ascir::TensorId, std::string> workspace_tensors;
  std::set<std::pair<std::string, std::string>> pre_api_extract_dup;

  std::string name;
  bool has_workspace_node{false};

  Tiler tiler;
  TPipe tpipe;
  Loop root_loop;

  explicit Kernel(const std::string& kernel_name);
  ~Kernel();

  static Status ParseGraph(const ascir::ImplGraph &graph, const ascir::FusedScheduledResult& fused_schedule_result, Kernel &kernel);
  Status Generate(const std::string &impl_graph_name, const std::string &tiling_data, std::string &result,
                  const ascir::ImplGraph &graph);
  static std::string GetIncludeApiHeaderFiles(const ascir::FusedScheduledResult &fused_schedule_result);
  static std::string IncludeAndDefines(const ascir::FusedScheduledResult &fused_schedule_result,
                                       const std::string &kernel_task_type, bool use_tensor_desc = false,
                                       bool is_inductor = false);
  std::string TilingKeyFuncDeclare(const std::string &impl_graph_name, const std::string &tiling_data) const;
  std::string GenTilingFuncCall(const std::string &impl_graph_name, const std::string &tiling_data, uint32_t index,
                                bool enable_group_parallel = false, bool need_sync_all = false) const;
  std::string GenTilingFuncCall(const std::string &impl_graph_name, const std::string &tiling_data) const;
  std::string GenCubeTilingFuncCall(const ascir::ImplGraph &impl_graph) const;
  std::string GenCubeTilingSingleFuncCall(const bool is_batch, const bool is_cv_fuse, bool is_bias,
                                          bool is_offset_w) const;
  ge::Status GenCubeCommonTiling(std::stringstream &ss, const bool is_batch) const;
  std::string GenCubeCommonTilingSingleFuncCall(const ascir::ImplGraph &impl_graph) const;
  static std::string KernelFuncDeclare(const std::string &graph_name,
                                       const ascir::FusedScheduledResult &fused_schedule_result,
                                       bool use_list_tensor = false, bool is_inductor = false);
  Status GlobalTensorInit(std::string &result) const;
  Status GlobalTensorAssign(std::string &result) const;
  Status GlobalTensorDefine(std::string &result) const;
  Status LocalTensorQueBufAlloc(std::string &result, const ascir::ImplGraph &graph) const;
  static Status GenKernelFuncByTilingKey(const ascir::FusedScheduledResult& fused_schedule_result,
                                         std::stringstream &ss, bool use_list_tensor = false,
                                         const CodegenConfig& config = {false, true},
                                         const std::string &kernel_task_type = "");
  void SetUseListTensor(bool use_list_tensor);
  Status ParseOptimizeInfo(const ascir::NodeView &node, const ascir::TensorView &tensor);
  Status ParseScalarNeedGenBlkTensors(const ascir::NodeView &node, ascir::TensorId id);
  Status OutputTensorIsUbScalar(const ascir::NodeView &node, bool &is_ub_scalar) const;
  Status GenerateKernelByNode(const ascir::ImplGraph &graph, std::stringstream &ss,
                              std::unordered_set<const std::string *> &kernel_file_ptr);
  Status GenerateMacro(std::stringstream &ss);
  Status GenerateSubGraphFuncDef(const Loop *loop, std::stringstream &ss) const;
  void SetUsingAttCalcQBTSizeConfig(bool using_att_calc_qbt_size);
  void SetEnableParallelCompile(bool enable_parallel_compile);
  bool GetEnableParallelCompile() const;
  Status GenerateVecFuncOfCVFusion(std::stringstream &result, bool vector_no_db_flag);
  Status InitCVFusionAddr(std::stringstream &result, bool vector_no_db_flag);
  static std::string GenKernelFuncCallForInductor(const ascir::FusedScheduledResult &fused_schedule_result);
  Status ParseUbScalarOptimizationInfo(const ascir::NodeView& node, Tensor& t, ascir::TensorId id, bool is_all_link_vf);
  Status JudgeIsLoadLinkStoreAndVec(const ascir::NodeView& node, Tensor& t, ascir::TensorId id);
 private:
  static std::vector<std::string> GenPackingFunctions(std::stringstream &ss_define,
                                                      const std::vector<Variable> &kernel_args,
                                                      const std::vector<std::vector<std::string>> &per_group_func_calls,
                                                      int64_t max_group_per_compile_unit,
                                                      uint32_t &function_id);
  static void GenPackingFunctionCalls(std::stringstream &ss,
                                      const std::vector<Variable> &kernel_args,
                                      const std::vector<std::string> &func_names);
  static std::string PackingFuncDeclare(const std::string &func_name, const std::vector<Variable> &kernel_args);
  static void AppendFuncCall(std::stringstream &ss, std::vector<std::vector<std::string>>::const_iterator begin,
                             std::vector<std::vector<std::string>>::const_iterator end, bool need_sync_all = true);
  static void AppendFuncCall(std::stringstream &ss,std::vector<std::vector<TilingFuncCall>> &per_group_func_calls,
                             std::vector<TilingFuncCall> &current, size_t depth, uint32_t &tiling_key, bool is_cube = false);
  static std::vector<Variable> PackingFuncArgs(const std::string &tiling_data_type,
                                               const ::ascir::FusedScheduledResult& fused_schedule_result,
                                               bool use_list_tensor);
  static void FakeTilingIds(std::stringstream &ss, uint32_t function_id_end);
  Status ParseWorkspaceTensor(const ascir::TensorAttr *tensor, const ascir::FusedScheduledResult& fused_schedule_result,
                              std::set<int64_t> &output_indices,
                              const std::unordered_map<ascir::TensorId, size_t> &output_tensorid_to_index,
                              const std::map<size_t, std::string> output_index_to_name);
  static Status GenSingleGroupKernelWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                                     const CodegenConfig& config, std::stringstream &ss,
                                                     std::stringstream &ss1, bool use_list_tensor);
  static Status GenMulGroupKernelWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                                  const CodegenConfig& config, std::stringstream &ss,
                                                  std::stringstream &ss1, bool use_list_tensor);
  static Status GenKernelFuncWithRegTilingKey(const ascir::FusedScheduledResult &fused_schedule_result,
                                              const CodegenConfig& config, std::stringstream &ss,
                                              std::stringstream &ss1, bool use_list_tensor);
  static Status GenSingleGroupKernelWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                        const std::vector<ge::AscGraph> &schedule_graphs,
                                                        const CodegenConfig& config, std::stringstream &ss,
                                                        std::stringstream &ss1, bool use_list_tensor,
                                                        std::unordered_set<const std::string *> &kernel_file_ptr);
  static Status GenMulGroupKernelWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                     const size_t graph_id, const CodegenConfig& config,
                                                     std::stringstream &ss, std::stringstream &ss1, bool use_list_tensor,
                                                     std::unordered_set<const std::string *> &kernel_file_ptr);
  static Status GenKernelFuncWithParseTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                                 const CodegenConfig& config, std::stringstream &ss,
                                                 std::stringstream &ss1, bool use_list_tensor);
  static int64_t GetMaxGroupPerCompileUnit(bool enable_parallel_compile);
  static Status GenCubeCommonFuncOfCVFusion(const ascir::FusedScheduledResult &fused_schedule_result,
                                            const size_t graph_id, const size_t common_index,
                                            const CodegenConfig &config, std::stringstream &ss, std::stringstream &ss1,
                                            const bool use_list_tensor,
                                            std::unordered_set<const std::string *> &kernel_file_ptr);
  static Status GenCubeCommonFuncForScheduleGroup(const ascir::FusedScheduledResult &fused_schedule_result,
                                                  const size_t graph_id, const size_t common_index,
                                                  const size_t group_index, const CodegenConfig &config,
                                                  std::stringstream &ss, std::stringstream &res_ss,
                                                  const bool use_list_tensor,
                                                  std::unordered_set<const std::string *> &kernel_file_ptr);

 private:
  static Status GenCubeCommonFuncForAIV(const ascir::FusedScheduledResult &fused_schedule_result, size_t graph_id,
                                        const size_t common_index, const size_t group_index,
                                        const CodegenConfig &config, std::stringstream &ss, std::stringstream &vec_ss,
                                        const bool use_list_tensor,
                                        std::unordered_set<const std::string *> &kernel_file_ptr);
  static Status GenCubeCommonFuncForAIC(const ascir::FusedScheduledResult &fused_schedule_result, size_t graph_id,
                                        const size_t common_index, const size_t group_index,
                                        const CodegenConfig &config, std::stringstream &ss, std::stringstream &cube_ss,
                                        const bool use_list_tensor,
                                        std::unordered_set<const std::string *> &kernel_file_ptr);
  static Status GenCubeCommonFuncForAICMix(const ascir::FusedScheduledResult &fused_schedule_result,
                                                   const size_t graph_id, const size_t common_index,
                                                   const size_t group_index, const CodegenConfig &config,
                                                   std::stringstream &ss, std::stringstream &cube_ss,
                                                   const bool use_list_tensor,
                                                   std::unordered_set<const std::string *> &kernel_file_ptr);
  std::map<std::string, size_t> input_name_to_index_;
  std::map<std::string, size_t> output_name_to_index_;
  bool use_list_tensor_ = false;
  bool enable_parallel_compile_ = true;
};
}

std::ostream &operator<<(std::ostream &os, const codegen::Code &obj);

Status LoopAxisDistance(const std::vector<ascir::AxisId> &current_loop,
                        const std::vector<ascir::AxisId> &node_sched_axis, const ascir::AxisId node_loop_axis,
                        int32_t &distance);
#endif

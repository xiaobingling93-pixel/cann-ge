# KvCacheExtParam

调用Pull或Push相关接口时传入的扩展参数。

```
struct KvCacheExtParam {
  std::pair<int32_t, int32_t> src_layer_range = {-1, -1};  // KV传输时源端的层数范围
  std::pair<int32_t, int32_t> dst_layer_range{-1, -1};  // KV传输时目的端的层数范围
  uint8_t tensor_num_per_layer = 2U;                       // KV传输时一层的tensor数量
  uint8_t reserved[127];                                   // 预留字段
}
```

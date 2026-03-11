# CacheDesc

Cache的描述信息

```
struct CacheDesc {
  CachePlacement placement = CachePlacement::kDevice;    // 内存类型
  uint32_t num_tensors = 0U;                             // Cache包含的tensor个数
  DataType data_type = DT_UNDEFINED;                     // Cache中tensor的数据类型
  std::vector<int64_t> shape;                            // Cache中tensor的shape
  uint8_t reserved[128];                                 // 预留
}
```

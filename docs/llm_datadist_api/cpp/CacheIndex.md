# CacheIndex

Cache的索引

```
struct CacheIndex {
  uint64_t cluster_id;        // cache所在的集群ID
  int64_t cache_id;           // cache的ID
  uint32_t batch_index;       // PullKvCache时用于指定batch的下标
  uint8_t reserved[128];      // 预留
}
```

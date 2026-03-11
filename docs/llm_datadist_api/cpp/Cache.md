# Cache

Cache，其中维护了一组tensor的地址

```
struct Cache {
  int64_t cache_id = -1;                     // Cache的ID
  std::vector<uintptr_t> tensor_addrs;       // Cache中各tensor的地址, 在单进程多卡场景中，多卡的地址依次排列。
  CacheDesc cache_desc;                      // Cache描述
  uint8_t reserved[128];                     // 预留
}
```

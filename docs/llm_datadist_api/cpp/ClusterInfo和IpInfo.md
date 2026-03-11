# ClusterInfo和IpInfo

用于描述集群信息，用于建链与断链。

```
struct ClusterInfo {
  uint64_t remote_cluster_id = 0U;     // 对端的LLM-DataDist的cluster_id
  int32_t remote_role_type = 0;        // 对端的LLM-DataDist的role_type，0表示全量，1表示增量
  std::vector<IpInfo> local_ip_infos;  // 本地LLM-DataDist的IP信息，详见如下结构体IpInfo
  std::vector<IpInfo> remote_ip_infos; // 对端LLM-DataDist的IP信息，详见如下结构体IpInfo
  uint8_t reserved[128];               // 预留
}

struct IpInfo {
  AscendString ip;         // IP地址
  uint16_t port = 0U;      // 端口号
  uint8_t reserved[128];   // 预留
}
```

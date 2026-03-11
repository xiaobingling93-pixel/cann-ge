# ClusterInfo和IpInfo（待废弃）

数据类型ClusterInfo和IpInfo定义在“$\{install\_path\}/latest/include/ge/llm\_engine\_types.h“。其中$\{install\_path\}为runtime安装路径，root用户的默认路径是“/usr/local/Ascend“。具体信息如下。

```
struct ClusterInfo {
  uint64_t remote_cluster_id = 0U;     // 对端的llm engine的cluster_id
  int32_t remote_role_type = 0;          // 对端的llm engine的role_type，0表示全量，1表示增量
  std::vector<IpInfo> local_ip_infos;   // 本地llm engine Device的IP信息
  std::vector<IpInfo> remote_ip_infos; // 对端llm engine Device的IP信息
}

struct IpInfo {
  uint32_t ip = 0U;   // 小端IP地址转换后的uint32_t的值
  uint16_t port = 0U; // 端口号，client侧不需要指定；server侧port需要指定(hccl指定一个固定port让用户填写)
}
```

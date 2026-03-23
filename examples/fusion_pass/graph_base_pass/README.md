## 融合Pass样例

本目录提供了继承GE提供的类并重写其方法来实现自定义融合pass的相关样例：

| 样例                              | 样例链接                                               |
|---------------------------------|----------------------------------------------------|
| MatMul+Add融合为GEMM自定义pass样例      | [README](1_fuse_matmul_add_pass/README.md)         |
| 移动Concat后ReLu至Concat前的自定义pass样例 | [README](2_move_relu_before_concat_pass/README.md) |
| 修改卷积算子data format的自定义pass样例 | [README](3_modify_conv_data_format_pass/README.md) |

## 开发指南

更多关于融合Pass开发的信息，请参考：[融合Pass开发指南](../融合Pass开发指南.md)

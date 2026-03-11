# DataFlow错误码

DataFlow错误码含义如下。

| 返回码 | 含义 | 解决方法 |
| --- | --- | --- |
| PARAM_INVALID=145000U | 参数校验无效 | 结合具体接口和日志找到不符合校验规则的参数，请按照资料或结合日志提示进行修改。 |
| SHAPE_INVALID=145021U | 输入tensor的shape异常 | 请结合日志修改输入的shape。 |
| DATATYPE_INVALID=145022U | 输入tensor的datatype异常 | 请结合日志修改输入的datatype。 |
| NOT_INIT=145001U | dataflow未初始化 | 请参照资料和样例代码在使用dataflow接口前先调用[init](dataflow-init.md)接口。 |
| INNER_ERROR=545000U | Python层内部错误 | 请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| FAILED=0xFFFFFFFF | C++层内部错误 | 请根据日志排查问题，或联系工程师处理（您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。）。<br>日志的详细介绍，请参见《日志参考》。 |
| SUCCESS=0 | 执行成功 | - |

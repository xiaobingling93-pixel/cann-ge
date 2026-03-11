# 多func处理函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

用户自定义的多flow func处理函数。

## 函数原型

- 普通场景下，函数输入由框架准备完毕后直接给用户使用，用户直接使用入参中的flowMsg即可。

    ```
    std::function<int32_t> (const std::shared_ptr<MetaRunContext> &runContext, const std::vector<std::shared_ptr<FlowMsg>> &flowMsg)
    ```

- 流式输入（即函数入参为队列）场景下，由用户自行从输入队列中获取数据使用。

    ```
    std::function<int32_t> (const std::shared_ptr<MetaRunContext> &runContext, const std::vector<std::shared_ptr<FlowMsgQueue>> &flowMsgQueue)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| runContext | 输入 | 处理函数的上下文信息。 |
| FlowMsg/FlowMsgQueue | 输入 | 函数的入参/输入队列。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

- 如果有不可恢复的异常信息发生，返回ERROR。
- 其他情况则调用SetRetcode设置输出tensor的错误码。
- 如果返回success，调度会终止。

## 约束说明

使用流式输入flow func时，需要在ProcessPoint编译配置文件中，将对应func的stream\_input字段设置为true。例如：

```
{
    "func_list": [
        {
            "func_name": "Func",
            "stream_input": true
        }
    ],
    "input_num": 1,
    "output_num": 1,
    "target_bin": "libfunc.so",
    "workspace": "./",
    "cmakelist_path": "CMakeLists.txt",
}
```

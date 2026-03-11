# RegProcFunc

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

注册多flow func处理函数，结合[MetaMultiFunc注册函数宏](MetaMultiFunc注册函数宏.md)来注册flow func。

## 函数原型

- 注册普通flow func，即flow func输入为flowMsg时使用。

    ```
    FlowFuncRegistrar &RegProcFunc(const char *flowFuncName, const CUSTOM_PROC_FUNC &func)
    ```

- 注册流式输入（即函数入参为队列）flow func，即flow func输入为flowMsgQueue时使用。

    ```
    FlowFuncRegistrar &RegProcFunc(const char *flowFuncName, const CUSTOM_PROC_FUNC_WITH_Q &func)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flowFuncName | 输入 | flow func的名称。不可以设置为NULL，必须以“\0”结尾。 |
| func | 输入 | 多flow func的处理函数，处理函数原型为<br>using CUSTOM_PROC_FUNC = std::function<int32_t(<br>T *, const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsg>> &)>;<br>或<br>using CUSTOM_PROC_FUNC_WITH_Q = std::function<int32_t(<br>T*, const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsgQueue>> &)>; |

## 返回值

返回当前的FlowFuncRegistrar类对象

## 异常处理

无。

## 约束说明

无。

# RegisterMultiFunc

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

注册多flow func。

不建议直接使用该函数，建议使用[MetaMultiFunc注册函数宏](MetaMultiFunc注册函数宏.md)来注册flow func。

## 函数原型

- 注册普通flow func，即flow func输入为flowMsg时使用。

    ```
    bool RegisterMultiFunc(const char *flowFuncName, const MULTI_FUNC_CREATOR_FUNC &funcCreator) noexcept
    ```

- 注册流式输入（即函数入参为队列）flow func，即flow func输入为flowMsgQueue时使用。

    ```
    bool RegisterMultiFunc(const char *flowFuncName, const MULTI_FUNC_WITH_Q_CREATOR_FUNC &funcWithQCreator) noexcept
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flowFuncName | 输入 | flow func的名称。不可以设置为NULL，必须以“\0”结尾。 |
| funcCreator/funcWithQCreator | 输入 | 多flow func的创建函数。 |

## 返回值

- true
- false

## 异常处理

无。

## 约束说明

无。

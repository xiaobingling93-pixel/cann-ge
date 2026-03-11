# dataflow.utils.generate\_deploy\_template

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据FlowGraph生成指定部署位置的option:"ge.experiment.data\_flow\_deploy\_info\_path"所需要的文件的模板。

## 函数原型

```
generate_deploy_template(graph, file_path)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| graph | FlowGraph | 构造完成的FlowGraph图。 |
| file_path | str | 生成用于指定部署信息的文件路径（包含文件名）。 |

## 返回值

无。

## 调用示例

```
import dataflow as df
#省略构图过程
dag = df.FlowGraph([output])
df.utils.generate_deploy_template(dag, "deploy_info.json")
```

## 约束说明

生成模板部署位置默认使用device0，如需调整需要对文件进行手动修改。

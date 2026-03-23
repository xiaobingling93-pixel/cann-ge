# 自定义融合pass开发指南

## 概述

自定义融合pass是GE提供的一种改图能力，关于GE相关内容可点击
[此处](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/graph/graphdevg/atlasag_25_0081.html)。
本指南介绍如何开发一个融合pass来实现自定义改图功能，
此外我们还提供了可运行的[sample](../fusion_pass)以供参考。
总体上，开发者通过继承GE提供的类并重写其方法来实现自定义融合pass，
接着调用注册宏将pass注册到指定阶段。

- 对于**一般**场景的融合操作，继承`PatternFusionPass`类实现自定义融合pass类，并通过`REG_FUSION_PASS`注册宏将pass注册到指定阶段。
- 对于**1对N**场景(单节点替换为N个节点)的融合操作，继承`DecomposePass`类实现自定义融合pass类，并通过`REG_DECOMPOSE_PASS`注册宏将pass注册到指定阶段。

两种场景在使用上大同小异，以下将进行详细介绍。

## 一般场景下的融合pass开发

本节首先对一般场景下涉及的核心数据结构`PatternFusionPass`作出介绍，
其次介绍开发过程中需要重写的3个函数:`Patterns`、`MeetRequirements`与`Replacement`，
最后介绍了如何将pass注册到指定阶段。

### PatternFusionPass

PatternFusionPass声明如下所示:

```c++
class PatternFusionPass : public FusionBasePass {
 public:
  Status Run(GraphPtr &graph, CustomPassContext &pass_context) override;
 protected:
  virtual std::vector<PatternUniqPtr> Patterns() = 0;
  virtual bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result);
  virtual GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) = 0;
};
```

`Run`函数调用`Patterns`获取模板的拓扑pattern，将pattern在目标graph中逐一匹配，
调用`MeetRequirements`对匹配到的pattern作出是否需要被替换的判断，
最后通过`Replacement`获取目标结构，将满足替换条件的pattern进行替换。

### 需要重写的3个函数：Patterns、MeetRequirements与Replacement

开发者通过继承`PatternFusionPass`并重写`Patterns`、`MeetRequirements`与`Replacement`方法实现自定义融合pass的开发：

| 函数             | 说明                                                         | 是否必须重写         |
| :--------------- | :----------------------------------------------------------- | :------------------- |
| Patterns    | 定义在目标图中匹配的模板拓扑。返回一个或多个图结构指针。     | 是                   |
| MeetRequirements | 对Patterns匹配到的图结构按条件进行过滤。输入匹配结果，返回布尔值。 | 否，默认直接返回true |
| Replacement      | 定义替换结构。输入匹配结果，返回图指针。                     | 是                   |

---

#### Patterns

`Patterns`定义在目标图中匹配的一个或多个模板拓扑，
使用`EsGraphBuilder`构建一张DAG图来表达pattern，如下所示：

```c++
std::vector<PatternUniqPtr> Patterns() override {
  std::vector<PatternUniqPtr> patterns;
  // 使用EsGraphBuilder构建pattern
  auto graph_builder = es::EsGraphBuilder("pattern");
  // 此处定义pattern
  // ...
  // 初始化Pattern对象
  auto graph = graph_builder.BuildAndReset({xxx});
  auto pattern = std::make_unique<Pattern>(std::move(*graph));
  patterns.emplace_back(std::move(pattern));
  // 可以继续向patterns中添加多个pattern
  // ...
  return patterns;
}
```

`EsGraphBuilder`为图构建器类，用于构建计算图。
该类提供了创建输入、变量、标量、向量等图元素的函数，以及设置图属性和构建的功能。
下表列出了`EsGraphBuilder`中用于创建图元素的部分常用函数：

| 成员函数       | 说明                                                         |
| :------------- | :----------------------------------------------------------- |
| CreateInput    | 创建图输入节点，可以创建具有指定名称和类型的图输入节点。     |
| CreateConst    | 创建Const算子，支持数据类型int64_t 、int32_t 、uint64_t 、uint32_t 、float。 |
| CreateVector   | 创建向量常量，支持数据类型int64_t 、int32_t 、uint64_t 、uint32_t 、float。 |
| CreateScalar   | 创建标量常量，支持数据类型int64_t 、int32_t 、uint64_t 、uint32_t 、float。 |
| CreateVariable | 创建变量，需要显示指定变量名。                               |
| ...            | ...                                                          |


推荐开发者使用Eager Style API进行pattern的定义，其提供了定义输入、常量与算子等接口，
以下是使用Eager Style API定义一个ReLu单算子pattern的示例：

```c++
std::vector<PatternUniqPtr> patterns;
auto graph_builder = es::EsGraphBuilder("pattern");
auto data = graph_builder.CreateInput(0);
auto relu = es::Relu(data);
auto graph = graph_builder.BuildAndReset({relu});
auto pattern = std::make_unique<Pattern>(std::move(*graph));
patterns.emplace_back(std::move(pattern));
```

> **注意！**
> 
> - 用于匹配的pattern需要满足**自包含**(除了边界的输出算子，边界内所有算子的数据输出消费者都要在边界内)，
> 非自包含的pattern不会被匹配。
> 
> - 自定义pass基于**当前图**匹配，对子图的修改需要在子图中再次调用pass。


除了上文中探讨的匹配图结构，我们还提供了两种接口实现对pattern更细粒度的定义：

##### CaptureTensor

定义过程中可以捕获pattern中一个tensor，从而在match_result中按序获取。
方法声明如下，入参node_output为`NodeIo`类型，由节点与索引组成，表示为某个节点的某个输出。

```c++
// CaptureTensor声明
Pattern &CaptureTensor(const NodeIo &node_output);

// NodeIo结构体
struct NodeIo {
  GNode node;
  int64_t index;
};
```

调用CaptureTensor捕获relu示例如下：

```c++
std::vector<PatternUniqPtr> patterns;
auto graph_builder = es::EsGraphBuilder("pattern");
auto data = graph_builder.CreateInput(0);
auto relu = es::Relu(data);
auto graph = graph_builder.BuildAndReset({relu});
auto pattern = std::make_unique<Pattern>(std::move(*graph));
// 调用CaptureTensor捕获relu
pattern->CaptureTensor({*relu.GetProducer(), 0})
patterns.emplace_back(std::move(pattern));
```

如何在match_result中获取tensor参见[后文示例](#target0)。

##### PatternMatcherConfig

构造自定义pass可以传入`PatternMatcherConfig`以使能const值匹配能力与ir属性及其值匹配能力。
基类`PatternFusionPass`构造函数如下：

```c++
explicit PatternFusionPass(std::unique_ptr<PatternMatcherConfig> match_config);
```

类`PatternMatcherConfigBuilder`提供两个函数作为匹配能力的开关，
`Build`函数用于构造`PatternMatcherConfig`。

- `EnableConstValueMatch`开启const值匹配，
在匹配过程中将对pattern中定义的const/constant进行值的匹配，值相等才认为匹配成功。

- `EnableIrAttrMatch`开启ir属性及其值匹配，
pass将在pattern匹配过程中对pattern中节点上携带的IR属性的数量和值进行匹配。

以下是名为`CustomFusionPass`的自定义pass类打开const值匹配的构造函数：

```c++
  explicit CustomFusionPass()
      : PatternFusionPass(PatternMatcherConfigBuilder().EnableConstValueMatch().Build()) {}
```

---

#### MeetRequirements

对于`Patterns`获取到的匹配结果，在`MeetRequirements`中进行筛选。
由上文`Run`函数的实现中可以看到，每个`MatchResult`类型的匹配结果作为`MeetRequirements`的入参，
通过`MatchResult`开发者可以获取匹配结果的信息进行筛选，
最后返回的布尔值作为是否替换该匹配结果的依据，如下所示：

```c++
bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
  // 可以使用传入的match_result对匹配结果进行筛选
  // 满足条件返回true
  if (IsSatisfy(match_result)) {
    return true;
  }
  // 不满足条件返回false
  return false;
}
```

<a id="target0"></a>
`MatchResult`是匹配结果类，包含匹配结果的节点、连边等信息。
开发者可以使用`MatchResult`成员函数获取匹配结果的相关信息以进行筛选，
以下是使用`GetCapturedTensor`校验ReLu输出是否为动态shape的示例：

  ```c++
  NodeIo relu_output;
  if(match_result->GetCapturedTensor(0,relu_output) != GRAPH_SUCCESS){
    return false;
  }
  TensorDesc relu_out_tensor_desc;
  relu_output.node.GetOutputDesc(relu_output.index, relu_out_tensor_desc);
  if (relu_out_tensor_desc.GetShape().GetShapeSize() != -1){
    return false;
  }
  return true;
  ```

---

#### Replacement

`Replacement`中定义目标结构，替换与`Patterns`中匹配且`MeetRequirements`为true的部分。
与`Patterns`一样，使用`EsGraphBuilder`定义结构，此处不再赘述：

```c++
GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
  auto replacement_graph_builder = es::EsGraphBuilder("replacement");
  // 此处定义替换结构
  // ...
  return replacement_graph_builder.BuildAndReset({r_a});
}
```

> **注意！**
> 如果pass注册阶段在InferShape后，需要在`Replacement`中自行调用`GeUtils::InferShape`，
> 此外如果要使用`GeUtils::CheckNodeSupportOnAicore`判断目标结构是否支持，该函数的调用需要在InferShape之后。

---

### 注册自定义融合pass

完成对融合pass的定义后，需要使用注册宏`REG_FUSION_PASS`将其注册到对应阶段，
如下是将名为`CustomFusionPass`的自定义pass注册到`kBeforeInferShape`阶段：

```c++
REG_FUSION_PASS(CustomFusionPass).Stage(CustomPassStage::kBeforeInferShape);
```

`CustomPassStage`是一个枚举类，可选值为：

| 枚举值                  | pass执行阶段                                           |
| :---------------------- | :----------------------------------------------------- |
| kBeforeInferShape       | InferShape前，注册在这个阶段的pass不需要自行infershape |
| kAfterInferShape        | InferShape后                                           |
| kAfterBuiltinFusionPass | 执行完内置融合pass后                                   |
| kInvalid                | 不生效                                                 |

---

## 1对N场景下的融合pass开发

1对N场景下的pass继承的基类为`DecomposePass`。
由于被替换结构是单个节点，此处pattern不再需要通过`Patterns`定义，
而是在构造函数中直接传入算子类型，如下所示：

```c++
class CustomOne2NPass : public DecomposePass {
 public:
  CustomOne2NPass(const std::vector<AscendString> &op_types) : DecomposePass(op_types) {}
```

`op_types`在注册pass阶段传入，
如下是使用注册宏`REG_DECOMPOSE_PASS`将`Conv2D`作为`op_types`初始化`CustomOne2NPass`，
并将其注册在`kAfterInferShape`：

```c++
REG_DECOMPOSE_PASS(CustomOne2NPass, {"Conv2D"}).Stage(CustomPassStage::kAfterInferShape);
```

---

与一般场景类似，继承自`DecomposePass`的pass也需要重写`MeetRequirements`与`Replacement`，
但两方法的入参类型不再是`MatchResult`而是`GNode`，即通过构造时传入的`op_types`在图中匹配到的节点。

```c++
bool MeetRequirements(const GNode &matched_node) override {
    ...
}
GraphUniqPtr Replacement(const GNode &matched_node) override {
    ...
} 	
```
# npu\_model

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

如果UDF部署在host侧，执行时数据需要从device拷贝到本地进行运算。对于PyTorch场景，如果计算全在device侧，输入输出也是在device侧，执行时数据需要从device拷贝到host，执行后PyTorch再将数据搬到device侧，影响执行性能，使用npu\_model可以优化为不搬移数据（即直接下沉到device执行）的方式触发执行。

## 函数原型

```
装饰器@npu_model
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| optimize_level | int | - 1：PyTorch场景下，通过UDF nn引擎完成输入输出数据下沉到device执行，默认值为1。<br>  - 2：把PyTorch模型编译成图，直接作为nn模型导出，优化为npu模型加载执行，需要配合input_descs使用。 说明： 该配置项在修饰类的时候起作用，修饰函数不能配置。 |
| input_descs | [[TensorDesc](dataflow-TensorDesc.md)] | 当optimize_level=2时，用于表达torch导出成图的输入tensor描述，示例如下：<br>input_descs=[TensorDesc(dtype = df.DT_INT64, shape = [2,1,4]),TensorDesc(dtype =<br>df.DT_FLOAT, shape = [2,1,4])],<br>当shape中某一维度为负值，表示输入是动态的，通过npu_model最终会导出成动态图。 |
| num_returns | int | 装饰器装饰函数时，用于表示函数的输出个数，不设置该参数时默认函数返回一个返回值。该参数与使用type annotations方式标识函数返回个数与类型的方式选择其一即可。 |
| resources | dict | 用于标识当前func需要的资源信息，支持memory、num_cpus和num_npus。memory单位为M; num_npus表示需要使用npu资源数量，为预留参数，当前仅支持1。例如：{"memory": 100, "num_cpus": 1, "num_npus": 1} |
| env_hook_func | function | 此钩子函数用于给用户自行扩展在Python UDF初始化之前必要的Python环境准备或import操作。 |
| visible_device_enable | bool | 开启后，UDF进程会根据用户配置num_npus资源自动设置ASCEND_RT_VISIBLE_DEVICES，调用get_running_device_id接口获取对应的逻辑ID，当前num_npus仅支持1，因此该场景下get_running_device_id结果为0。 |

## 返回值

正常场景下返回被装饰的函数。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
@df.npu_model(optimize_level=1)
class FakeModel1(nn.Module):
    def __init__(self):
        super().__init__()

    # 模拟模型推理
    @df.method()
    def forward(self, input_image):
        return F.interpolate(input_image, size=(256, 256), mode='bilinear')

@df.npu_model(optimize_level=1, input_descs=[df.TensorDesc(dtype=df.DT_FLOAT, shape=[1, 3, 768, 768])])
class FakeModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = 0.5
        self.std = 0.5

    # 模拟模型推理
    @df.method()
    def forward(self, input_image):
        return (input_image - self.mean) / self.std

@df.npu_model()
def preprocess(input_image):
    # 模拟图片裁切
    transform = transforms.Compose([transforms.CenterCrop(512)])
    return transform(input_image)

@df.npu_model()
def postprocess(input_image):
    mean = 0.5
    std = 0.5
    img = input_image * std + mean
    return F.interpolate(img, size=(512, 512), mode='bilinear')
```

## 约束说明

- 需安装对应Python版本的torch\_npu包。
- 输入输出必须为npu tensor。
- 一组输入对应一组输出，不支持流式输入输出。

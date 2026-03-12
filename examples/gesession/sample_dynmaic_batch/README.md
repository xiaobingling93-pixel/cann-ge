# 样例使用指导

## 功能描述

该样例主要是基于Onnx ResNet-50网络（单输入、单Batch）实现多batch场景下图片分类的功能。

在该样例中：
1.  先使用样例提供的脚本transferPic.py，将2张\*.jpg图片都转换为\*.bin格式，同时将图片从1024\*683的分辨率缩放为224\*224。
2.  解析resnet50模型，然后直接对2张图片（batch值为2）进行推理，得到推理结果，再对推理结果进行处理，输出top5置信度的类别标识。


## 目录结构

```
├── data
│   ├── dog1_1024_683.jpg               // 测试数据,需要按指导获取测试图片,放到data目录下
│   ├── dog2_1024_683.jpg               // 测试数据,需要按指导获取测试图片,放到data目录下

├── model
│   ├── resnet50_Opset16.onnx           // Onnx ResNet-50网络的模型文件

├── scripts
│   ├── build.sh                        // sample编译脚本
│   ├── run.sh                          // sample运行脚本
│   ├── transferPic.py                  // 将*.jpg转换为*.bin，同时将图片从1024*683的分辨率缩放为224*224

├── src
│   ├── CMakeLists.txt                  // 编译配置脚本
│   ├── sample_dynamic_batch.cpp        // 主函数，图片分类功能的实现文件

├── CMakeLists.txt                      // 编译脚本，调用src目录下的CMakeLists文件
```

## 环境要求

- 通过安装指导 [环境准备](https://gitcode.com/cann/ge/blob/master/docs/build.md#2-%E5%AE%89%E8%A3%85%E8%BD%AF%E4%BB%B6%E5%8C%85) 正确安装`toolkit`和`ops`包

- 设置环境变量（假设包安装在`/usr/local/Ascend/`）

    ```
    source /usr/local/Ascend/cann/set_env.sh
    ```

## 实现步骤

1.  以运行用户登录开发环境。

2.  下载代码并上传至环境后，请先进入根目录下"examples/gesession/sample_dynmaic_batch"样例目录。

    请注意，下文中的样例目录均指"examples/gesession/sample_dynmaic_batch"目录。

3.  准备ResNet-50模型。
    1.  获取ResNet-50原始模型。

        您可以从以下链接中获取ResNet-50网络的模型文件，并以运行用户将获取的文件上传至开发环境的"样例目录/model"目录下。如果目录不存在，需要自行创建。

        -   ResNet-50网络的模型文件（\*.onnx）：单击[Link](https://github.com/onnx/models/blob/main/Computer_Vision/resnet50_Opset16_timm/resnet50_Opset16.onnx)下载该文件。

4.  准备测试图片。
    1.  请从以下链接获取该样例的输入图片，并以运行用户将获取的文件上传至开发环境的"样例目录/data"目录下。如果目录不存在，需自行创建。

        [https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1\_1024\_683.jpg](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg)

        [https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog2\_1024\_683.jpg](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog2_1024_683.jpg)

    2.  切换到“样例目录/data“目录下，执行transferPic.py脚本，将\*.jpg转换为\*.bin，同时将图片从1024\*683的分辨率缩放为224\*224。在“样例目录/data“目录下生成2个\*.bin文件。

        ```
        python3 ../scripts/transferPic.py
        ```

        如果执行脚本报错“ModuleNotFoundError: No module named 'PIL'”，则表示缺少Pillow库，请使用**pip3 install Pillow --user**命令安装Pillow库。

## 构建验证

1.  以运行用户登录开发环境。

2.  请先进入根目录下"examples/gesession/sample_dynmaic_batch"样例目录。

    请注意，下文中的样例目录均指"examples/gesession/sample_dynmaic_batch"目录。

3.  切换到"样例目录/scripts",编译程序。

    ```
    bash build.sh
    ```

4.  运行程序
    ```
    bash run.sh
    ```

5.  执行结果

    执行成功后，在屏幕上的关键提示信息示例如下，提示信息中的index表示类别标识、value表示该分类的最大置信度，这些值可能会根据版本、环境有所不同，请以实际情况为准：

        [INFO] SAMPLE start to execute.
        [INFO] Initialize ge success
        [INFO] Set device 0 success
        [INFO] Parse model ../model/resnet50_Opset16.onnx success
        [INFO] Graph add success
        [INFO] Graph compile success
        [INFO] Start to process file:../data/dog1_1024_683.bin
        [INFO] Start to process file:../data/dog2_1024_683.bin
        [INFO] Graph run success
        [INFO] Result of picture 1:
        [INFO] top 1: index[162] value[xxxxxx]
        [INFO] top 2: index[161] value[xxxxxx]
        [INFO] top 3: index[166] value[xxxxxx]
        [INFO] top 4: index[167] value[xxxxxx]
        [INFO] top 5: index[163] value[xxxxxx]
        [INFO] Result of picture 2:
        [INFO] top 1: index[267] value[xxxxxx]
        [INFO] top 2: index[266] value[xxxxxx]
        [INFO] top 3: index[265] value[xxxxxx]
        [INFO] top 4: index[153] value[xxxxxx]
        [INFO] top 5: index[99] value[xxxxxx]
        [INFO] Output data success
        [INFO] SAMPLE PASSED

    **说明：**
    类别标签和类别的对应关系与训练模型时使用的数据集有关，本样例使用的模型是基于imagenet数据集进行训练的，您可以在互联网上查阅imagenet数据集的标签及类别的对应关系。
    当前屏显信息中的类别标识与类别的对应关系如下：
    "161": \["basset", "basset hound"\]、
    "267": \["standard poodle"\]。

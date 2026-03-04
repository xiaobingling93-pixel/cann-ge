# 样例使用指导

## 功能描述

该样例主要是基于Onnx ResNet-50网络（单输入、动态多Batch）实现图片分类的功能。

在该样例中：
1.  先使用样例提供的脚本transferPic.py，将2张\*.jpg图片都转换为\*.bin格式，同时将图片从1024\*683的分辨率缩放为224\*224。
2.  加载离线模型om文件，对2张图片（batch值为2）进行推理，得到推理结果，再对推理结果进行处理，输出top5置信度的类别标识。

在加载离线模型前，提前将Onnx ResNet-50网络的模型文件转换为适配昇腾AI处理器的离线模型。

## 目录结构

```
├── data
│   ├── dog1_1024_683.jpg               // 测试数据,需要按指导获取测试图片,放到data目录下
│   ├── dog2_1024_683.jpg               // 测试数据,需要按指导获取测试图片,放到data目录下

├── model
│   ├── resnet50.om                     // Onnx ResNet-50网络的模型文件,需要按指导获取atc转换后的om文件,放到model目录下

├── scripts
│   ├── build.sh                        // sample编译脚本
│   ├── run.sh                          // sample运行脚本
│   ├── transferPic.py                  // 将*.jpg转换为*.bin，同时将图片从1024*683的分辨率缩放为224*224

├── src
│   ├── acl.json                        // 系统初始化的配置文件
│   ├── CMakeLists.txt                  // 编译配置脚本
│   ├── sample_resnet50_imagenet_classification_dynamic_batch.cpp       // 主函数，图片分类功能的实现文件

├── CMakeLists.txt                      //编译脚本，调用src目录下的CMakeLists文件
```

## 环境要求

- 已完成[昇腾AI软件栈在开发环境上的部署](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)

## 实现步骤

1.  以运行用户登录开发环境。

2.  下载代码并上传至环境后，请先进入根目录下"examples/acl/2_sample_resnet50_imagenet_classification_dynamic_batch"样例目录。

    请注意，下文中的样例目录均指“examples/acl/2_sample_resnet50_imagenet_classification_dynamic_batch”目录。

3.  准备ResNet-50模型。
    1.  获取ResNet-50原始模型。

        您可以从以下链接中获取ResNet-50网络的模型文件，并以运行用户将获取的文件上传至开发环境的"样例目录/model"目录下。如果目录不存在，需要自行创建。

        -   ResNet-50网络的模型文件（\*.onnx）：单击[Link](https://github.com/onnx/models/blob/main/Computer_Vision/resnet50_Opset16_timm/resnet50_Opset16.onnx)下载该文件。

    2.  将ResNet-50原始模型转换为适配昇腾AI处理器的离线模型（\*.om文件）。

        切换到样例目录，执行如下命令(以Atlas A2系列产品为例)：

        ```
        atc --model=resnet50_Opset16.onnx --framework=5 --output=resnet50_dynamic_batch --soc_version=Ascend910B1 --input_format=NCHW --output_type=FP32 --input_shape="x:-1,3,224,224" --dynamic_batch_size="1,2,4,8"
        ```

        -   --model：原始模型文件路径。
        -   --framework：原始框架类型。0：表示Caffe；1：表示MindSpore；3：表示TensorFlow；5：表示ONNX。
        -   --soc\_version：昇腾AI处理器的版本。版本获取可参考[Link](https://hiascend.com/document/redirect/CannCommunityAtcSocVersion)。
        -   --output\_type：指定输出的数据类型为float32。
        -   --output：生成的resnet50.om文件存放在“样例目录/model“目录下。建议使用命令中的默认设置，否则在编译代码前，您还需要修改sample\_resnet50\_imagenet\_classification.cpp 中的omModelPath参数值。
        -   --input\_shape: 指定输入数据的shape值，其中不想指定的维度可以将其设置为-1。
        -   --dynamic\_batch\_size: 设置动态batch_size参数。

            ```
            const char* omModelPath = "../model/resnet50_dynamic_batch.om";
            ```

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

2.  请先进入根目录下"examples/acl/2_sample_resnet50_imagenet_classification_dynmaic_batch"样例目录。

    请注意，下文中的样例目录均指"examples/acl/2_sample_resnet50_imagenet_classification_dynmaic_batch"目录。

3. 设置环境变量，配置程序编译依赖的头文件与库文件路径。

    设置以下环境变量后，编译脚本会根据"{DDK_PATH}环境变量值/include/"目录查找编译依赖的头文件，根据{NPU_HOST_LIB}环境变量指向的目录查找编译依赖的库文件。

    **注意**，在配置{NPU_HOST_LIB}环境变量时，需使用的"devlib"目录下*.so库，确保在编译基于AscendCL接口的应用程序时，不依赖其它组件（例如Driver）的*.so库，编译成功后，运行应用程序时，系统会根据LD_LIBRARY_PATH环境变量查找“Ascend-cann-toolkit安装目录/lib64”目录下的*.so库，同时会自动链接到所依赖的其它组件的*.so库。

    -   配置示例如下所示：

        ```
        # toolkit默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
        export DDK_PATH=/usr/local/Ascend/cann
        export NPU_HOST_LIB=$DDK_PATH/devlib
        # toolkit指定路径安装，${install_path}为toolkit的安装目录。
        export DDK_PATH=${install_path}/cann
        export NPU_HOST_LIB=$DDK_PATH/devlib
        ```

4.  切换到"样例目录/scripts",编译程序。

    ```
    bash build.sh
    ```

5.  运行程序
    ```
    bash run.sh
    ```

6.  执行结果

执行成功后，在屏幕上的关键提示信息示例如下，提示信息中的index表示类别标识、value表示该分类的最大置信度，这些值可能会根据版本、环境有所不同，请以实际情况为准：

        [INFO] acl init success
        [INFO] open device 0 success
        [INFO] create context success
        [INFO] create stream success
        [INFO] load model ../model/resnet50_dynamic_batch.om success
        [INFO] start to process file:../data/dog1_1024_683.bin
        [INFO] start to process file:../data/dog2_1024_683.bin
        [INFO] model execute success
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
        [INFO] output data success
        [INFO] SAMPLE PASSED

**说明：**
类别标签和类别的对应关系与训练模型时使用的数据集有关，本样例使用的模型是基于imagenet数据集进行训练的，您可以在互联网上查阅imagenet数据集的标签及类别的对应关系。
当前屏显信息中的类别标识与类别的对应关系如下：
"161": \["basset", "basset hound"\]、
"267": \["standard poodle"\]。

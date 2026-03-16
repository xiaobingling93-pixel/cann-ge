# GE资料书架总览

## 文档

- [图模式开发指南](https://hiascend.com/document/redirect/CannCommunityGraphguide)

  面向单卡的图编译和执行，提供GE基本概念、原理介绍、以及如何使用GE图引擎接口进行图的构建，编译和运行等。

- [DataFlow开发指南](https://hiascend.com/document/redirect/CannCommunityDataflow)

  面向异构和集群的图编译和执行，介绍如何通过DataFlow接口构建、修改、编译和执行计算图。
  
- [LLM DataDist开发指南](https://hiascend.com/document/redirect/CannCommunityLLMDatadistdev)

  面向大模型，介绍如何使用LLM-DataDist接口实现集群间的数据传输，构建大模型推理分离式框架。

## 技术文章

  - [计算图优化](https://www.hiascend.com/zh/developer/techArticles/20240621-1)

    介绍GE如何通过通用的图优化技术（比如常量折叠）和特有的增强图优化技术（比如Shape优化技术），提升算法计算效率。

  - [多流并行](https://www.hiascend.com/zh/developer/techArticles/20240701-1)

    介绍多流并行技术的实现原理和使能方式，以及通过该技术，如何提高硬件资源利用率。

  - [内存复用](https://www.hiascend.com/zh/developer/techArticles/202407005-1)

    介绍GE如何结合业界标准的内存优化手段，利用全图视角精细调优内存复用算法和拓扑排序，进一步压缩网络内存占用，从而降低网络内存占用。

  - [模型下沉](https://www.hiascend.com/zh/developer/techArticles/20240715-1)

    介绍GE如何通过图模式的Host调度和模型下沉调度的方式，提升模型调度性能，缩短模型E2E执行时间。

  - [动态shape图调度加速](https://www.hiascend.com/zh/developer/techArticles/20250911-1)

    介绍Host调度优化的关键技术，以及如何通过这些技术提升异构系统资源的利用率。

  - [自动融合](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/graph/graphguide/autofuse_1_0001.html)

    介绍自动融合的实现原理和使能方式，以及通过该技术，如何缩短模型E2E时间。

## **API**

-   [GraphEngine API](./graph_engine_api/README.md)
-   DataFlow API
    -   [C++接口](./dflow_api/README_cpp.md)
    -   [Python接口](./dflow_api/README_python.md)
-   LLM DataDist API
    -   [C++接口](./llm_datadist_api/README_cpp.md)
    -   [Python接口](./llm_datadist_api/README_python.md)


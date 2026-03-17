# 自定义融合pass开发指南

## 概述

自定义融合pass是GE提供的一种改图能力，关于GE相关内容可点击
[此处](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/graph/graphdevg/atlasag_25_0081.html)。
本指南介绍如何开发一个融合pass来实现自定义改图功能，
此外我们还提供了可运行的样例以供参考,
其中graph_base_pass[sample](../fusion_pass/graph_base_pass)是通过graph接口来实现自定义融合pass，
而pattern_base_pass[sample](../fusion_pass/pattern_base_pass)是继承GE提供的类并重写其方法来实现自定义融合pass，
总体上，推荐开发者通过继承GE提供的类并重写其方法来实现自定义融合pass。
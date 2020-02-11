---
title: Clementine教程
date: 2019-11-07 10:42:24
categories: 数据挖掘
tags:
- Clementine
copyright: true
---

# Clementine教程

数据挖掘课程要求使用这个软件Clementine来进行实验，之前完全没听说过这个软件。网上搜到的资料也比较少，特别是CSDN上面有个博客名字叫做Clementine完整教程，然后内容也是Clementine教程这几个字，把我给惊呆了，这也能写博客？现在实验都已经做完了因此记录下使用方法，希望对其他人能有帮助。

<!--more-->

## 简介

### 页面

在Clementine软件中只需要简单的像画图一样，把整个数据挖掘的操作流程对应的节点给连接起来就可以使用了。先对页面简单的介绍下，页面如下。

![](https://res.cloudinary.com/bravey/image/upload/v1573095648/blog/clementine.jpg)

在最下面的一栏是相对应的各个节点：Favorites中是常用的节点，Sources中是输入的数据节点，Field中是数据的属性设置节点，Modeling中是一些数据挖掘模型节点比如决策树神经网络等模型，Output中是输出节点用来查看数据。

在右上角中的Streams显示整个工作台中的所有Stream，Outputs中是每次运行后得到输出结果，Models中是流程所用到的模型。

数据挖掘的过程可以简单的分为两步：第一步对模型进行训练 第二步对训练好的模型进行使用。而一个完整的数据挖掘流程Stream中在Clementine中需要包含：数据输入节点->输入属性设置节点->模型节点->输出节点。在第一步模型训练中可以不使用输出节点。

## 数据导入

在下方节点栏中的Sources中选择Var.file节点，拖到中间的画布。然后双击设置输入数据的文件导入，如下所示：
![](https://res.cloudinary.com/bravey/image/upload/v1573097265/blog/varfile1.jpg )

在导入数据后选择分隔符我的数据是用tab分割的)，之后点击Apply完成设置。

## 数据显示

在导入了数据后可以使用输出节点来输出数据，选择Outputs中的Table节点来输出数据。如下所示：

![](C:\Users\BraveY\AppData\Local\Temp\1573097747688.png)

对输入节点训练集.txt邮件有个connect可以进行连接。或者使用快捷键F2来连接。连接好后右键execute就可以得到上图显示了。

## 模型训练

以使用决策树模型为例子，给定的输入数据集是一个在线测试系统中学生做的各项测试的数据，final是期末考试是否及格。也就是通过学生在测试系统的数据，来预测学生期末考试是否会及格。在选择模型前，需要在Fidel Ops中选择一个type节点来设置输入数据中各个属性。如下所示：

![](https://res.cloudinary.com/bravey/image/upload/v1573098328/blog/type.jpg )

第一个PersonId设置为Typeless不进行输入，决策树输出的是final，并设置为flag类型。其他数据都设置为range。

接下来选择模型，在Modeling中选择C5.0这个决策树模型。并与前面的type节点连接。如下所示：

![](https://res.cloudinary.com/bravey/image/upload/v1573098962/blog/train_tree.jpg)

可以选择专家模式自定义剪枝率等参数。设置好后执行就可以得到训练好的模型了，将会显示在右上角的model中，这样就完成模型训练这个Stream了。

在右上角的models中选择训练好的final模型，右键browse，然后选择viewer就可以查看训练好的决策树模型了。如下所示。

![](https://res.cloudinary.com/bravey/image/upload/v1573099406/blog/model.jpg)
![](https://res.cloudinary.com/bravey/image/upload/v1573099392/blog/viewer.jpg)

### Apriori

补充Apriori模型的例子，整体的架构为
![](https://res.cloudinary.com/bravey/image/upload/v1581417277/blog/Clementine/Apriori_model.jpg)

数据的导入设置为：

![](https://res.cloudinary.com/bravey/image/upload/v1581416895/blog/Clementine/data.jpg)

type节点的设置为：

![](https://res.cloudinary.com/bravey/image/upload/v1581416896/blog/Clementine/type.jpg)

模型的设置为：

![](https://res.cloudinary.com/bravey/image/upload/v1581416895/blog/Clementine/model_setting.jpg)

其中支持度和置信度这些参数是可以自己调的。

## 模型使用

同样导入数据节点验证集.txt，设置好type节点(测试集的flag属性也设置为输入了)，再把训练好的模型从右上角给拖进来，最后设置输出节点。整个流程与前述相同，得到的验证流程如下：
![](https://res.cloudinary.com/bravey/image/upload/v1573099826/blog/test_stream.jpg)
输出节点使用混淆矩阵来查看模型的准确性，具体设置为：
![](https://res.cloudinary.com/bravey/image/upload/v1573099827/blog/confuse_matrix.jpg)
最后执行就可以得到最终的显示了如下：
![](https://res.cloudinary.com/bravey/image/upload/v1573099826/blog/matrix_out.jpg)

## 总结

其他模型比如神经网络等的使用流程大同小异，如果需要数据集与测试集以及英文教程可以联系我。
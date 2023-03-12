---
title: LARGE BATCH OPTIMIZATION FOR DEEP LEARNING TRAINING BERT IN 76 MINUTES
date: 2020-06-01 09:55:32
categories: 论文阅读
tags:
copyright: true
---
## 来源

ICLR 2020

## 关键词

## 摘要

Large Batch加速训练的方法LARS，在注意力机制模型比如BERT上表现不好，性能提升在各个任务中表现不一致。本文首先研究一种有原则的分层自适应策略，以使用Large mini-batches来加快深度神经网络的训练。使用这种策略，本文开发了一种称为**LAMB**的新的分层自适应大批量优化技术。本文提供LAMB以及LARS的**收敛分析**，表明算法可以在一般非凸设置下的收敛到固定点。

LAMB在各种任务（例如BERT和RESNET-50训练）中的表现出色，而**超参数调整却非常少**。在BERT训练中，本文的优化程序可以使用非常大的32868批量，而不会降低性能。通过batch size增加到TPUv3 Pod的内存限制，BERT训练时间可以从3天减少到仅76分钟。 LAMB实现公布在[网上]([https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py) )

## 引言

随着大规模数据集的出现，即使使用诸如梯度梯度下降（SGD）等计算有效的优化方法来训练大型深度神经网络也变得特别具有挑战性。即训练大型模型的时间变得非常长。

本文的**目标**是研究和开发优化技术，以加速训练大型深度神经网络，其中主要集中在基于SGD变体的方法上。SGD的可扩展性受到其固有顺序性的限制。由于存在这种局限性，在深度学习背景下改善SGD培训时间的传统方法主要采用**分布式异步**设置，但是，由于异步引入的隐式过时（stalenes）限制了方法的并行化，通常会导致性能下降。简单地增加batch size 虽然可以增加速度，但是会降低泛化性能并降低计算收益。

Large mini-batch上的同步SGD受益于SGD中使用的随机梯度变化的减少，从而可以在SGD中使用更大的学习率，根据batch size的具体大小**线性调整学习率**可以进一步加快训练。但是线性缩放调整学习率的**缺点**：（i）在初始阶段，学习率的线性缩放是有害的；因此，最初需要使用手动调整的缓慢增加学习速度的预热策略，并且（ii）学习速度的线性缩放在超过一定的batch size之后也是有害的。

最近提出了使用**分层自适应学习率**的SGD变体来解决此问题，比如非常出名的LARS算法。但该算法的性能在跨任务上不一致，比如BERT上表现糟糕。此外LARS的适应方法，还缺少很多理论分析。

**本文贡献**

1. 在LARS的启发下，本文研究了专门针对大批量学习的通用适应策略，并为该策略提供了直觉
2. 基于适应策略，本文开发了一种新的优化算法（LAMB），以实现SGD中学习率的适应性。此外，本文为LARS和LAMB都提供了收敛分析，以在非凸设置中达到固定点。
   本文重点介绍了将这些方法用于大批量设置的好处。
3. 本文展示了LAMB在多个挑战性任务中的强大性能。使用LAMB，本文将训练BERT的批量大小扩展到32k以上，而不会降低性能。时间从3天减少到76分钟。本文的工作是将BERT训练时间减少到几个小时以内的第一项工作。
4. 本文还展示了LAMB训练像RESNET这样的最新图像分类模型的效率。本文是第一个可以为RESNET-50达到SOTA的自适应求解器，因为像Adam这样的自适应求解器无法获得这些任务的SGD精度。

---

## 方法

### 预备知识

完整的推导见论文，

本文首先将优化问题定义为：非凸随机优化问题

$$
\min _{x \in \mathbb{R}^{d}} f(x):=\mathbb{E}_{s \sim \mathbb{P}}[\ell(x, s)]+\frac{\lambda}{2}\|x\|^{2}
$$

SGD方法是解决上述问题最简单一阶算法：

$$
x_{t+1}=x_{t}-\eta_{t} \frac{1}{\left|\mathcal{S}_{t}\right|} \sum_{s_{t} \in \mathcal{S}_{t}} \nabla \ell\left(x_{t}, s_{t}\right)+\lambda x_{t}
$$

对于large batch b = T并使用适当的学习率，对于SGD的迭代，有：

$$
\mathbb{E}\left[\left\|\nabla f\left(x_{a}\right)\right\|^{2}\right] \leq O\left(\frac{\left(f\left(x_{1}\right)-f\left(x^{*}\right)\right) L_{\infty}}{T}+\frac{\|\sigma\|^{2}}{T}\right)
$$

在实践中很难调整SGD中的学习速率，尤其是在大批量设置中。此外，对$ L_{\infty}$的依赖（尺寸上最大的平滑度）可能会导致收敛速度显着降低。

### 算法

#### 常规算法

一个基础的优化算法(SGD,ADAM)的迭代策略是：

$$
x_{t+1}=x_{t}+\eta_{t} u_{t}
$$

$ u_{t}$是t时间步的更新量，文章建议对Large batch进行两项更改。

1. 更新量使用l2正则，即使用$u_t/\|u_t\|$,逐层完成的。
2. 学习率使用函数$\phi\left(\left\|x_{t}\right\|\right)$ 进行调整。也是逐层完成。

修改后的SGD更新策略：

$$
x_{t+1}^{(i)}=x_{t}^{(i)}-\eta_{t} \frac{\phi\left(\left\|x_{t}^{(i)}\right\|\right)}{\left\|g_{t}^{(i)}\right\|} g_{t}^{(i)}
$$

#### LARS 算法

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-01 151610.jpg)

文章对LARS算法提供了收敛分析

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-01 152422.jpg)

#### LAMB 算法

伪代码：

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-01 152156.jpg)

与LARS算法不同：LAMB的适应性有两个方面：（i）关于ADAM中使用的第二个时刻的平方根的每维归一化；以及（ii）由于分层适应性而获得的分层归一化。

收敛性证明：

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-01 152507.jpg)

## 实验

模型：BERT和RESNET-50

参数设置 $\beta_1 = 0.9,\beta_2=0.999$，只微调了学习率。多项式衰减的学习率：$\eta_{t}=\eta_{0} \times(1-t / T)$,在batch size增大的时候没有再次微调超参数。使用LR缩放规则的平方根来自动调整学习率和线性epoch的warm up。

实验平台：TPUv3 Pod。

使用了grid search 来调整学习方法ADAM，ADAGRAD，LARS等的超参数。

### BERT训练

数据集：与BERT原文的一致，主要关注SQuAD任务。使用F1分数作为指标，对比的模型是BERT 论文中的baseline。

训练过程：除了修改优化函数LAMB为，其余均一致。

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-01 162220.jpg)

对LAMB使用了Mixed-Batch 训练

BERT训练的第一阶段：batch size限制到65536，因为进一步加大并没有更好的加速效果。第二阶段重新对学习率进行warm up。

**与ADAMW 和LARS的比较**

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-01 163643.jpg)

### IMAGENET上的RESNET-50训练

各种优化函数的对比

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-01 171012.jpg)

### 超参数调整

自动调整的学习率：

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-01 171256.jpg)

## 讨论

---

## 思考

1. 本文是第一项对BERT的优化工作，以往的工作主要针对ResNet等视觉模型，对于新推出的GPT-3有没有相关工作？

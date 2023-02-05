---
title: Extremely Large Minibatch SGD:Training ResNet-50 on ImageNet in 15 Minutes
date: 2020-06-02 16:00:28
categories: 论文阅读
tags:
- Large Batch
copyright: true
---

## 摘要

本文通过将minibatch size提高到32k实现了在ImageNET上用15分钟训练完ResNet50。为了保证精读使用了RMSprop warm-up,batch normalization，以及slow-start learning rate schedule.

## 引言

训练深度神经网络在计算上是昂贵的。更高的可扩展性（更大的数据集和更复杂的模型）和更高的生产率（更短的培训时间以及更快的试验和错误）要求通过**分布式计算**来加速。
本文证明，在不影响准确性经过精心设计的软件和硬件系统的情况下，使用large batch可以进行高度并行的训练。

------

## 方法

主要使用了《Accurate, large minibatch SGD: training ImageNet in 1 hour》文章的方法。

### RMSprop Warm-up

本文发现主要的挑战是训练开始时的优化难度。为了解决这个问题，本文从RMSprop [7]开始训练，然后逐步过渡到SGD。

momentum SGD 和RMSprop 的结合
$$
\begin{aligned}
m_{t} &=\mu_{2} m_{t-1}+\left(1-\mu_{2}\right) g_{t}^{2} \\
\Delta_{t} &=\mu_{1} \Delta_{t-1}-\left(\alpha_{\mathrm{SGD}}+\frac{\alpha_{\mathrm{RMSprop}}}{\sqrt{m_{t}}+\varepsilon}\right) g_{t}, \text { and } \\
\theta_{t} &=\theta_{t-1}+\eta \Delta_{t}
\end{aligned}
$$
从RMSprop 开始，然后切换到SGD(需要平滑切换，突然切换有副作用)

对$\alpha_{SGD}$的调度，类似与ELU激活函数：
$$
\alpha_{\mathrm{SGD}}=\left\{\begin{array}{ll}
\frac{1}{2} \exp \left(2\left(\mathrm{epoch}-\beta_{\mathrm{center}}\right) / \beta_{\mathrm{period}}\right) & \left(\mathrm{epoch}<\beta_{\mathrm{center}}\right) \\
\frac{1}{2}+2\left(\mathrm{epoch}-\beta_{\mathrm{center}}\right) / \beta_{\mathrm{period}} & \left(\mathrm{epoch}<\beta_{\mathrm{center}}+\frac{1}{2} \beta_{\mathrm{period}}\right) \\
1 & (\text { otherwise })
\end{array}\right.
$$
$\alpha_{SGD}$先指数增长，之后到达$\beta_{center}$epoch后达到1/2,之后线性增长到$\beta_{center} + \frac{1}{2}\beta_{period}$ epoch, 超参数$\beta_{center}=10,\beta_{period}=5$

### Slow-Start Learning Rate Schedule

为了进一步克服最初的优化困难，本文使用略微修改的学习率时间表，以延长初始阶段并降低初始学习率

初始学习率
$$
\eta_{base} =0.1 \cdot \frac{b_{\text {botal }}}{256}=0.1 \cdot \frac{n b_{\text {local }}}{256}
$$
$b_{local}$ 是一个worker的batch size。

SGD前40个epoch用$0.5\cdot\eta_{base}$,随后的30个epoch为$0.075\cdot\eta_{base}$,接下来的15个epoch $0.01\cdot\eta_{base}$,最后5个为$0.001\cdot\eta_{base}$

### Batch Normalization without Moving Averages

随着batch size的增加，均值和方差的batch normalization会移动平均值导致实际均值和方差的不准确估计。为了解决这个问题，本文只考虑了最后的minibatch，而不是移动平均数，并且对这些统计数据使用全归约通信来获得验证之前所有Workers的平均数。

## 结果

**软件**：使用Chainer和ChanierMN Chainer是一个开源的深度学习框架，具有“按运行定义”方法。 ChainerMN是Chainer的附加软件包，可实现具有同步数据并行性的多节点分布式深度学习。作为底层的通信库，本文使用了NCCL 2.0.5版和Open MPI 1.10.2版。

使用半精度浮点数来减少通信开销，对模型准确的影响较小。

**硬件**： 使用MN-1内部集群。包含128个节点。

**训练时间**

通信和迭代时间

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-02 175529.jpg)

![](C:\Users\BraveY\Documents\BraveY\blog\images\论文阅读\批注 2020-06-02 175659.jpg)

## 讨论

------

## 思考

1. 使用的软件不常用可以替换为Pytorch？
2. 比较的软件框架，硬件都不一致，是否有说服力？有没有统一平台的benchmark？




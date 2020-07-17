---
title: Measuring and Benchmarking Power Consumption and Energy Efficiency
date: 2020-07-14 16:05:52
categories: 论文阅读
tags: 
- 能耗
- SPEC SERT
copyright: true
---

## 来源

ICPE 18

## 关键词

功率，能效，性能，基准，测量，负载水平，SPEC

## 摘要

1. 显存能耗测量方法不考虑多个负载级别和工作负载组合。
2. 介绍了PTDaemon 能耗测量工具和Chauffeur 能耗评测框架
3. SPEC SERT 包含的工作负载，并介绍行业标准的计算效率基准

## 引言

1. 如何精确测量能耗？
2. 数据中心的冗余机制（灾难备份）导致的额外负载需要被考虑
3. SPEC 能耗方法学多重负载级别下进行能耗效率测评
4. 功率测量是在SPEC PTDaemon中实现的，它与功率分析仪和温度传感器进行通信。工作负载分派，结果收集和测试执行由Chauffeur框架处理。

------

## 方法

### 遵守的原则

1. 可重现
2. 公平
3. 可验证
4. 可使用

### 评测电力要求

1. 均方根电力测量
2. 每秒通过通讯接口将测量值记录到外部设备上
3. 不确定性少于1%
4. 定期校准至国家标准
5. 设备配置控制和程序界面记录
6. 能够处理安培数峰值（波峰因数），以在恶劣的功率条件下实现正确的读数

考虑环境温度

### PTDaemon

基于TCP-IP的公共接口集成到基准线束中，对评测软件隐藏不同硬件接口的协议和行为

![](https://res.cloudinary.com/bravey/image/upload/v1594956254/blog/paper/批注_2020-07-15_095058.jpg)

通过PTDaemon与传感器进行交互（TCP/IP），设备类型由守护程序初始调用时在命令行上本地传递的参数指定。

SUT和PTDaemon之间的通信协议独立于特定的功率测量设备类型,可以独立于要支持的测量设备来开发基准。

PTDaemon实现：

- 主进程：控制初试化网络命令界面
- 单独线程：管理功耗仪与温度传感器

为了支持不同的功率分析仪，每个受支持的设备都需要在PTDaemon中拥有自己的模块。周期性的更新各个模块以增加设备支持。可以根据手册来自己提交设备支持。

### 能耗效率方法学

**校准步** ：被测机器上的目标负载最大事务率？记为100%。负载的多层次测量。

负载层次定义为最大应用**程序吞吐量的百分比**，而不是目标CPU利用率。CPU利用率在不同核心和系统上差异较大。使用吞吐量百分比更加精确

#### 设备安装

两个物理系统：

- 控制系统：运行线束，报告器，并与外部测量设备连接。
- 被测机：运行负载

通过网络进行同步，数据收集

![](https://res.cloudinary.com/bravey/image/upload/v1594956254/blog/paper/批注_2020-07-15_105031.jpg)

每个逻辑CPU生成一个客户端，事务负载在客户端上单独运行。通过运行多个客户端实现并行性。

至少一个功耗仪与温度传感器。温度传感器确保实验环境一致。

### 负载和工作量

多个worklet构成一个workload，一个worklet是小规模的单元（事务）

### 阶段和间隔

worklet的三个阶段：

- warmup：不记录测量值，避免瞬时影响
- 校准：得到负载的最大层次
- 测量：实际测量

 每个阶段一个或多个间隔，用于阶段配置工作执行。

每个间隔包含前测量期与后测量期pre-measurement and a post-measurement period.，事务被在目标层次被执行，但不记录测量值。15s

测量阶段在两个时期之间，120s

切换负载，10s的缓冲期避免影响。间隔顺序执行，渐进测量顺序（多层次性能负载的测量 ）

事务吞吐率：校准阶段结果和目标负载层次的百分比决定

![ ](https://res.cloudinary.com/bravey/image/upload/v1594956254/blog/paper/批注_2020-07-15_111230.jpg)

使用每个间隔的能耗与吞吐率的平均值计算间隔的能耗效率

![](https://res.cloudinary.com/bravey/image/upload/v1594956254/blog/paper/批注_2020-07-15_111713.jpg)

## 实现

### Chauffeur

框架支持在多种负载下测量能耗。

SPEC PTDaemon与功率分析仪和温度传感器连接，Chauffeur对PTDaemon进行必要的调用，以便在适当的时间间隔内收集数据。

多种用途下运行的能力。与ssh2008相同原则。

**可伸缩性** ：多进程与多线程，在各种服务器上提供伸缩性。支持多节点运行

**易用** ： 自动配置，自动验证结果

**便携性** Java实现，跨系统

**灵活性** 灵活的改变运行时行为，（收集不同数据格式）通过配置文件，XML，HTML，TXT，CSV格式进行报告

### Chauffeur WDK 测量任意负载

代码编写逻辑

使用框架简化了测试逻辑，只用专注测试逻辑。

自动对负载进行多层次测试

开发一个worklet两个组件：（代码逻辑）

- Transaction事务 ：worklet测试的业务逻辑 两个方法
  - 产生输入（事务的随机数据）
  - 处理（接受输入然后处理并得到结果）
- User用户：获取状态信息

支持协同worklet，多个worklet同时工作。IO写入模拟

### SERT 和负载

最基础的配置要求：一个功耗仪，温度传感器，SUT，Controller

使用Chauffeur框架。控制安装在Controller上的软件，处理能耗记录的后勤工作。

SUT从Chauffeur实例（Director）接收指令来执行负载集合。负载集合由Worklet组成，Worklets是实际的代码，旨在强调特定的一个或多个特定系统资源，例如CPU，内存或存储IO。

每个功率分析仪和温度传感器均与SPEC **PTDaemon的专用实例进行交互**，该SPEC PTDaemon实例在执行Worklet时收集其读数。

**报告器**收集整理数据产出HTML，XML等格式的输出

评测的挑战：复杂的配置。（格式错误等等）

**系统配置发现**

SERT通过自动硬件发现过程和易于使用的GUI工作流程来解决这些问题，该流程可帮助用户生成高质量的准确报告。GUI减轻了测试配置，执行和报告编辑的负担，因此用户可以**专注于获得结果**。

![](https://res.cloudinary.com/bravey/image/upload/v1594956254/blog/paper/批注_2020-07-15_152831.jpg)

**定义和执行**

SERT由worklets套件组成，每个worklet从具体放方面测试被测机。LUworklet：CPU密集型的矩阵分解。序列IOworklet执行序列化IO操作。

不同方面的**负载**：CPU，内存和存储IO。

![](https://res.cloudinary.com/bravey/image/upload/v1594956254/blog/paper/批注_2020-07-15_153316.jpg)

Worklet连续执行，运行在自己的JVM或进程中避免干扰。每个JVM固定到特定处理器，以避免人为限制扩展。

多个JVM来运行单个worklet避免软件瓶颈限制伸缩性，主要是测量硬件的能耗不是软件的堆栈。

**Worklet**

12个

![](https://res.cloudinary.com/bravey/image/upload/v1594956254/blog/paper/批注_2020-07-15_155322.jpg)

**CPU**

数据压缩，加密/解密，复数算法，矩阵分解，浮点数组处理，排序算法，字符串处理

**存储IO**

读与写的事务

**内存**

使用预计算和缓存的数据查找来进行XML文档的操作和验证，以及对数据转换的四个主要类别进行具有读/写操作的数组操作；

**主动怠速**

稳定阶段，空闲阶段

没有网络IO的worklet,由配置修改器处理

### SPECpower_ssj2008

略过

## 结果

刻画了SPEC Power的方法学

SPEC PTDaemon可实现功耗的精确测量，而Chauffeur框架可实现工作负荷的调度，布置，执行和结果收集。
可以在ChauffeurWDK中使用PTDaemon和Chauffeur，以实现和测试研发工作负载的能源效率。
另一方面，SPEC SERT已经提供了大量可用于研究和服务器评级的工作负载。



----

## 思考

1. PTDaemon Chauffeur，SERT，ssj2008之间的关系？[参考](http://trickmore.blogspot.com/search/label/SERT) [参考2](https://zhuanlan.zhihu.com/p/45518506)
   
   1. ssj是基础，SERT通过PTDaemon
   
2. 支持比较多的功耗仪，定期更新（优点），配置性高

3. transaction?怎么翻译？ 事务一个工作流程，或模块

   1. 一次数据压缩就是一个transaction
   
   




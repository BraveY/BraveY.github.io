[NSQ](https://nsq.io/)是一个使用Go开发的分布式消息队列，对于学习Go语言和消息队列而言都是一个非常不错的开源项目，因此记录下学习NSQ的一些笔记。

# 简介

官网上NSQ宣传的特点包括

1. 分布式——去中心化，无单点故障，推模型来降低消息延迟
2. 可扩展——可水平扩展
3. 运维友好——容易配置和部署，提供预编译和Docker镜像版本
4. 易集成——支持主流语言的客户端，以及HTTP接口发送消息
   实际简单体验下来，的确部署和启动都比较快速方便，不需要提前做大量的配置就可以简单启动demo。

作为消息队列，NSQ的消息语义保证有：

1. 消息默认非持久化。消息保存在内存中，超过存储水位后会临时存到磁盘中。
2. 消息传递最少一次。这也意味着使用者自己去处理消息重传后导致的幂等问题。
3. 接受到的消息是无序的。主要是因为各个节点之前没有共享数据造成的，官方建议使用者自己使用延迟窗口来进行排序。
4. 消费者最终一定会发现所有生产者。

# 架构设计

## Topic

与Kafka的Topic设计做一个简单的对比，方便更好的学习。
NSQ中和Kafka一样使用Topic来组织消息也主要是发布定于模式，但使用推模型发送消息，而Kafka是基于拉模型的。因此实时性上NSQ会更好一些。
NSQ中Topic也会实际的存储消息，并且与Channels中的消息是单独队列来存储。而Kafka中的Topic主要是逻辑概念，实际的数据存储是由分区来存储消息的。
NSQ中每个Topic有一到多个的Channel来存储消息，每个Channel都会复制Topic中的消息。而Kafka中同一个Topic下的不同分区的消息是不同的。
NSQ中每个Channel 也会由多个消费者相连接。

下图展示了消息的轨迹。可以看到消息先是存储在topic中，然后广播到其对应的每个Channel中，最后被每个Channel中的消费者竞争消费。
![](https://lh6.googleusercontent.com/TYmXeG3t2yP5r_jogQ4x0iwpfjrAMiDynZXVnS0oLdM6H5xQFFknDjFlRyvKgJ4VUPFoVgJCkklRZHlMLZdqlmf-g_hMR3W6go88j-4Qzy3aJM9gDiRXv8gb1W-2JM6zP0IoPr5YleJoapjpEyTsoope=s2048)

## 节点发现

`nsqd`是消息处理节点，每个 `nsqd`节点都可以发送Topic相关的消息。但在分布式部署的模式下，发送消息之前需要先与 `nsqlookupd`建立TCP长连接周期性心跳更新状态，这样其他 `nsqd`节点才可以发现当前节点以及当前节点上的Topic对应的消息。

此外 `nsqlookupd`节点也作为中间件来实现生产者和消费者的解耦，生产者与消费者通过 `nsqlookupd`来发现彼此。
![](https://lh4.googleusercontent.com/H8QIzJBGl9_UF0DzYHVLY_SXKh-9HdJd4q5L0m09lttRMPqGPrtAOszrUv9ZJCOlj3rvH9Pd99yDu7Fom6n4r4eOZ6rp9ciZxFrrfpJLLWyJ8YjEivjY2VSnPA5h6_7bNUFmv2nh8yr8Sr0KdGT2EYUi=s2048)
官方建议 `nsqlookupd`最少部署两个来提供服务的可用性。
消费者通过向 `nsqlookupd`发送HTTP请求来获得可用的nsqd节点和节点上的Topic，Channels中的消息。
![](https://lh4.googleusercontent.com/VFxHQOjPPfRDagMHdDu6C-4pf7p18h3BldsqmRCsZ7eE6MnkGHHQ19hE_7OX3OVEWENSvnQpGNNWR38zUYYUHZNc1iSXL0129YFMSrEVp8eedCdrPzvxbXUUlGsWuhtzv9UMGkCELrF212436mMEwiB5=s2048)

## 单点故障消除

因为每个消费者与所有提供相应Topic的 `nsqd`节点连接，所以不会出现单点故障。某个 `nsqd`节点出现故障也不会影响服务。（这里自己有个疑问，消息是会在所有节点上保存副本吗？不然单点故障了消息岂不就丢失了？）
![](https://media.tumblr.com/tumblr_mat85kr5td1qj3yp2.png)

## 消息内存存储

消息默认使用内存存储，只有当超过水位的时候，会将消息存储到磁盘中。
这里有一个深度的概念就是指队列中消息的积压深度。当深度超过阈值的时候，消息就转存到磁盘中。

当将深度设置的很低的时候，比如1或者0，那么每个消息都会落盘，可以通过这种方式确保预期外重启的时候的消息不会丢失。
![](https://media.tumblr.com/tumblr_mavte17V3t1qj3yp2.png)

## 效率

为了大幅提升性能和消息吞吐率，NSQ设计的数据协议使用推模型来发送消息。
![](https://media.tumblr.com/tumblr_mataigNDn61qj3yp2.png)
消费者在与 `nsqd`建立连接并订阅Topic后，通过将自己设置为RDY状态来接受消息，同时设置对应RDY状态的值来接受指定的数量的消息。比如图中RDY 2则只会收到两条消息。通过这个方式来灵活的调节消费者的消费能力。这也提供了指定客户端进行调度的方式。

## 概览

`nsqd`向 `nsqlookupd`注册服务，而消费者通过 `nsqlookupd`来发现服务。之后消费者和所有 `nsqd`建立连接。
![](https://lh3.googleusercontent.com/2g3cuQekUnRvzxFPTaa-KPdvB9KyIV3ygljm4ZPJ_4cW0m-j2jEPYutKzP9DO_KFutHfsl2Os1D6lJpBJFVO81SXe4NbPqiDYOwXPGPRTL_oW5Uhhi3enEkN2KX0dniIIIMisxsO16LKMWNQy928C_hL=s2048)

# DEMO

启动demo的方式参见官方的[教程](https://nsq.io/overview/quick_start.html) 不需要额外设置相应的配置就能快速启动了，同时使用 `nsqadmin`来提供前端管理服务。本文不赘述。

# 总结

使用《数据密集型应用系统设计》一书中对 发布/订阅模式的流处理系统的分类方法对NSQ进行总结，可以比较好的了解NSQ的设计逻辑。书中的方法为提出如下两个问题进行分类。

1. 如果生产者发送消息的速度比消费者快所能处理的快，会发生什么？
   1. 一般而言由三种选择：系统丢弃消息；将消息缓存在队列中；使用流量控制。NSQ的方式是消息缓存在队列中，超过队列阈值后写入磁盘。相比而言Kafka是直接写入磁盘。
2. 如果节点崩溃或者暂时离线，是否会有消息丢失？
   1. 这一点如果NSQ将转存磁盘的阈值设置为0，那么所有消息都会落盘，因此不存在消息丢失。但是对于内存中的消息是如何保证不丢的，这个还需要后续进一步深入了解。

# 参考资料

https://nsq.io/overview/design.html
https://docs.google.com/presentation/d/1e9yIm-0aNba_H1gX_u7D1Qe5VWeU26FCjg7EEzoUztE/edit#slide=id.g3c47333ca6_0_0
https://blog.lpflpf.cn/passages/nsqd-study-1/

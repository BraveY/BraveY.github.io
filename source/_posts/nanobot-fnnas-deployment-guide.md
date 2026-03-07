---
title: 闲置笔记本变身边AI助手：飞牛NAS + Docker + 旁路由部署nanobot全攻略
date: 2026-03-07 10:00:00
tags:
  - NAS
  - Docker
  - nanobot
  - AI
  - 飞牛
categories:
  - 技术教程
---

## TL;DR

不需要macmini，使用一个闲置笔记本改造的飞牛NAS，在上面部署本地安全可控龙虾的AI助手。通过使用docker运行nanobot，配合阿里云Coding Plan的模型订阅，同时走虚拟机的旁路由科学上网，从而低成本的实现All in one的龙虾部署。

## openclaw部署方案选型

为了尝鲜爆火的openclaw， 从硬件、 软件和模型订阅来分析下选型。

### 硬件选型

硬件上选择主要是3个方案：

1. 购买mac mini单独部署。
    
	1. 优点：体验应该是最好。
    
	2. 缺点：额外购买mac mini成本太高。 不适合垃圾佬。
    
4. 闲置笔记本改造
    
	1. 优点：成本最低。 部署自己可控，硬件配置性能足够。
    
	2. 缺点：配置稍微麻烦。
    
7. 厂商云助手方案， 比如[阿里云的分钟级别部署openclaw](https://www.aliyun.com/activity/ecs/clawdbot?userCode=6rucfacw)
    
	1. 优点：配置最简单，成本中等。
    
	2. 缺点：掌控性较差，机器性能较差，内存配置比较小。
    

### 软件部署选型

软件上则是是否需要考虑安全隔离， 因为大模型的幻觉可能导致误删重要文件或者数据。如果要安全隔离则是使用docker容器部署，或者虚拟机部署。

docker容器部署的方案：

1. 优点：较强的隔离，AI助手无法方案操作到重要数据。快速配置和部署不需要手动安装各种环境依赖，性能损失少。
    
2. 缺点：容器重建后，应用数据不存在，需要重新配置应用环境。（只重启不需要重新配置环境，同时挂载磁盘可以持久化重要数据）
    

虚拟机部署的方案：

1. 优点： 完全的隔离，不担心任何安全问题。
    
2. 缺点： 性能损失较大。对硬件配置要求比较高。
    

### AI助手选型

可以参考这个回答[openclaw之后又出现了nanobot，它好用吗？](https://www.zhihu.com/question/2002042418774706034)

我这里选择nanobot是因为配置了一下openclaw的确步骤较多，稍微麻烦些。而nanobot非常精简，配置简单，4000行的代码方便学习和debug。对中文社区支持也比较好，默认支持钉钉作为channel，因为上班用钉钉所以选择这个。

### 模型订阅

模型订阅上只能选择订阅制，因为龙虾AI助手的token消耗量很大，所以使用订阅制成本才可控。至于用国内还是国外的模型则根据个人喜好进行了。国内厂商的模型订阅价格低，但是模型能力略差，不过基本也够用。

我选择的是阿里云的Coding plan 。首月7.9非常划算，lite版本的量普通用户完全够用。可以用我的[邀请链接](https://www.aliyun.com/benefit/ai/aistar?userCode=6rucfacw&clubBiz=subTask..12413312..10263)还能有10元优惠卷使用。

### 我的选择

因为我之前已经有一个闲置笔记本部署的飞牛NAS，所以硬件选择使用闲置的笔记本来进行部署。同时因为NAS本来就是全天候开机的，所以天然适合部署这种AI助手。

而软件部署方案上一开始打算使用虚拟机进行操作，但是飞牛NAS安装的虚拟机重启后就无法开机，折腾很久也没有搞好。 而且启动后发现CPU和内存占用比较大。宿主机CPU利用率30%起步，而内存更是基本上分配多少用多少，虚拟机分配的8G内存基本全部吃掉。所以体验很差，因此切换到docker来部署，体验很好， 非常快速。CPU基本不消耗，内存消耗也控制的不错。

![](https://picx.zhimg.com/80/v2-b59b228401954e6bcfc2f23046a55525_1440w.png?source=ccfced1a)


## 部署方案

### 飞牛NAS部署方案

笔记本的配置为4C8G的小新。使用网线直连的路由器，同时还有一个无线网卡。

![](https://picx.zhimg.com/80/v2-a484321ce74d09fdda0b7bedf19539ca_1440w.png?source=ccfced1a)


具体的飞牛NAS的部署教程可以参考[官方教程](https://help.fnnas.com/articles/v1/start/install-os.md)，这里不赘述， 但是注意使用网线连接路由器来方便后面虚拟机部署旁路由。

安装好飞牛NAS后， 我们需要安装飞牛应用中心的虚拟机和docker这两个应用。虚拟机用来安装旁路由做科学上网，docker用来部署nanobot。

### 旁路由部署

因为龙虾相关的很多技能资源都是在外网，所以如果不能科学上网，用起来很难受。 Linux的Node 22安装都依赖外网，基本上不能科学上网就不能安装openclaw。

我使用的是旁路由方案，相关知识可以参考[如何更好地使用旁路由](https://doc.istoreos.com/zh/guide/istoreos/practice/BypassRouter.html)。

使用有线网卡所以可以路由器桥接后开启ovs，用虚拟机进行安装旁路由。具体使用虚拟机安装旁路由的教程可以参考[飞牛iStoreOS旁路由虚拟机](https://club.fnnas.com/forum.php?mod=viewthread&tid=26481)。 跟着教程走来开启旁路由还是比较顺利的。

iStoreOS安装好了之后可以通过[are u ok](https://github.com/AUK9527/Are-u-ok)来下载对应的科学上网组件。 注意安装后openclash是在左侧的服务栏选型中，我第一次安装找了好一会。

![](https://picx.zhimg.com/80/v2-bd5bf89b4187c6f021d06e8b6580987b_1440w.png?source=ccfced1a)


科学上网配置好之后，家里相关需要科学上网的设备需要手动操作把wifi的ip设置从DHCP修改为静态分配，路由地址和DNS地址修改为旁路由的IP地址才行，不是自动生效的。如下是mac的wifi设置参考

![](https://picx.zhimg.com/80/v2-ac070e051cea9ca8ff9dba20f2808ffd_1440w.png?source=ccfced1a)

### 容器部署nanobot

这部分网上没有比较好的飞牛NAS部署教程，而且需要设置docker使用指定无线网卡来科学上网，所以我详细写下：

**容器配置初始化**

docker 的镜像仓库中搜索smanx/nanobot这个镜像。飞牛的镜像仓库能够直接拉取，不需要科学上网。

![](https://pic1.zhimg.com/80/v2-2a01c84c636df7a63e6317d90610c8d2_1440w.png?source=ccfced1a)


镜像下载后创建容器， 并挂载host的目录来持久化nanobot的配置。
直接点击下一步，这个容器只使用来初始化配置。

![](https://picx.zhimg.com/80/v2-f46dc885d2cd0d6c0e40d74957ca9ebd_1440w.png?source=ccfced1a)

重要：将nas的一个文件夹路径配置为挂载到容器/root/.nanobot目录。
我将宿主机的目录/vol1/1000/Docker/Nanobot/.nanobot挂载到了/root/.nanobot

![](https://picx.zhimg.com/80/v2-058b27581e583b14fb2c499eb27cccc8_1440w.png?source=ccfced1a)

环境变量中配置OPENAI兼容的模型厂商url

![](https://picx.zhimg.com/80/v2-f912ea9bcad811e67008b5b313bad99c_1440w.png?source=ccfced1a)


剩余的网络选择默认的bridge即可， 没有额外需要设置的了，点击下一步，然后勾选创建后启动，点击启动就行。

![](https://pica.zhimg.com/80/v2-407e61cd5ce30fc4c446ad171093c250_1440w.png?source=ccfced1a)

这样容器就会创建出来，并自动根据配置的环境变量来初始化配置启动nanobot。
可以查看运行日志是否配置完成，gateway是否启动成功。

![](https://pic1.zhimg.com/80/v2-64b1ff776d320f12eb564bca64026028_1440w.png?source=ccfced1a)

这样创建出来的容器重启后会卡住，因为镜像里面的启动步骤主要是两个步骤：

1. 配置config.json
	`nanobot onboard`
2. 启动网关
	`nanobot gateway`

因为我们的nanobot的工作目录/root/.nanobot是挂载到宿主机的，所以重启后对于已经存在配置，会要求交互式的选择是否覆盖已有的配置目录。因此重启后会一直卡在第一步骤，导致网关没有启动成功。

上文首次创建并启动的容器，我们只是用来初始化/root/.nanobot这个工作目录的配置文件结构。当在宿主机文件中已经存在对应的相关目录后，就说明配置已经初始化完成了。

![](https://picx.zhimg.com/80/v2-95492a18db5cfa2b14ecdddfb1a82be3_1440w.png?source=ccfced1a)

配置成功后，我们直接将容器关闭后执行删除， 后面改为使用docker compose来进行部署。

**docker compose 部署**

docker compose部署的好处是

1. 可以重置entrypoint 从而只执行nanobot gateway这个命令， 只启动网关。
2. 支持自动重启，保证服务的稳定性。

在部署之前需要将笔记本的无线网卡打开，然后将网关地址和dns地址设置为上面的旁路由ip地址。这样子nas上的无线网卡的就能实现科学上网了。

![](https://picx.zhimg.com/80/v2-f8657481e49a039cca4885b2b9f9a00a_1440w.jpg?source=ccfced1a)

接着我们需要创建一个容器网络，这个容器网络需要使用无线网卡作为出口，从而实现科学上网。具体设置教程可以参考这个[Docker网络指定使用的网卡作为出流量 | 码上星辰的技术札记](https://www.xiaozhuhouses.asia/article/jb45rscv/)

有了容器网络后，配置如下的docker compose yaml文件。其中mynet就是设置的容器网络。dns需要显示的设置为旁路由的DNS地址才能在容器中科学上网。（小龙虾安装后帮我排查出来的，不然依然无法科学上网）

```yaml
services:
  nanobot:
    image: smanx/nanobot:latest
    container_name: nanobot
    restart: unless-stopped
    environment:
      - PUID=0
      - PGID=0
      - TZ=Asia/Shanghai
      - NANOBOT_DEFAULT_MODEL=qwen3.5-plus
      - OPENAI_API_BASE=https://coding.dashscope.aliyuncs.com/v1
      - OPENAI_API_KEY=sk-sp-xxxxx
    volumes:
      - /vol1/1000/Docker/Nanobot/.nanobot:/root/.nanobot
    entrypoint: nanobot gateway
    ports:
      - 18790:18790
    networks:
      - mynet
networks:
  mynet:
    external: true 
dns:  # 旁路由DNS
  - 192.168.31.83
```

配置的操作页面如下：

![](https://picx.zhimg.com/80/v2-44f31b4350934fa648be7c4a9b286151_1440w.png?source=ccfced1a)

如此容器相关的设置就完成了，点击构建并启动容器就可以复用宿主机上的nanobot配置文件了。
容器启动后可以去容器页面查看是否正常运行。

### 通道配置

上面启动的nanobot没有配置对应的channel，只能在命令行里面手动对话。我们需要配置对应的通道，才能在聊天应用里面进行对话。

我使用的是钉钉， 可以[参考官方教程](https://open.dingtalk.com/document/dingstart/build-dingtalk-ai-employees)。大概的配置步骤如下：

- 需要用个人账号去创建一个组织，
- 在[开放平台](https://open-dev.dingtalk.com/)里面创建应用并配置对应的机器人。
    
	- 机器人的权限配置对应的允许单聊。
    
	- 机器人的消息接收模式使用Stream模式
    
- 发布应用。
	- 注意发布应用需要在最下面的版本管理与发布中点击创建新版本，这样才能顺利发布，钉钉才能搜索到机器人。
    
- 配置钉钉到nanobot    
	- 将钉钉机器人的id和秘钥填写到nanobot的配置文件中。下面的staffid可以先填写星号，如果后续发送失败了，再看日志里面具体的id是啥。
    
- 重启容器。
	- 配置好之后需要重启容器，配置才生效。
    
- 手机钉钉开始对话
	- 手机钉钉里面去搜索钉钉机器人，添加到群聊里面后，可以点击头像进行单聊。

这样我们的ai助手就算可以用了。

![](https://pica.zhimg.com/80/v2-c1317fe0ce33dd122830602f2978595c_1440w.jpg?source=ccfced1a)

## 技能推荐

因为配置了科学上网所以可以无缝使用clawdhub等外网技能资源。推荐下面三个：

- [agent-browser](https://github.com/vercel-labs/agent-browser) 给agent用的浏览器。可以截图，点击浏览器等。
- [skill-creator](https://github.com/anthropics/skills/tree/main/skills/skill-creator) 用来创建技能。
- [tavily](https://www.tavily.com/) 用来做网页搜索的mcp，每个月1000次的免费额度。可以替换brave搜索。

其他的技能就自己按需探索即可。
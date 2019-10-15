---
title: ceph 部署文档
date: 2018-04-17 17:36:03
categories: deploy
tags: ceph
---

# ceph 部署文档

<!--more-->

------

# 1.配置所有节点

## 创建ceph用户

## 安装配置NTP

1. `systemctl enable ntp`  ubuntu 14.04不可用，感觉已经安装过了，因此跳过。

## 配置hosts文件

`172.16.1.93 object1`
`172.16.1.94 object2`
`172.16.1.95 object3`
`172.16.1.66 object4`
`172.16.1.92 controller`

------

# 2. 配置ssh服务器

修改ssh的配置文件

Host controller
        Hostname gd92
        User cephuser
Host object1
        Hostname gd93
        User cephuser
Host object2
        Hostname hw101
        User cephuser
Host object3
        Hostname gd95
        User cephuser
Host object4
        Hostname gd66
        User cephuser

生成密钥并拷贝到4个osd节点上，无需拷贝到controller节点

------

# 3.安装ceph

主要参考链接：这些链接的操作大都一致，部分的顺序会有变化。

https://linux.cn/article-8182-1.html#4_10238

https://blog.csdn.net/styshoo/article/details/55471132

https://blog.csdn.net/styshoo/article/details/58572816

## 部署监控节点出现的问题

`ceph-deploy mon create-initial`

1. ` ceph-mon --cluster ceph --mkfs -i gd92 --keyring /var/lib/ceph/tmp/ceph-gd92.mon.keyring`

   问题：ceph.conf的配置文件中的`public network=172.16.1.92/24` 掩码前面多打了空格

   修改后重新执行命令，并加上`--overwrite-conf` 

2. [info]Running command: ceph --cluster=ceph --admin-daemon /var/run/ceph/ceph-mon.controller.asok mon_status

   `admin_socket: exception getting command descriptions: [Errno 2] No such file or directory`

   似乎是ceph -deploy 的问题，或者是ubuntu14.04的问题。教程是ubuntu16.04的

   此问题非hostname 不对应

   非conf 不同步导致。--overwrtie-conf  无作用。

   解决办法：按照14.04方法重新安装ceph-deploy

## 部署osd节点出现的问题

1. 使用`ceph-deploy disk list ceph-osd1 ceph-osd2 ceph-osd3`检查磁盘可用性时报错，使用`ceph-deploy osd prepare ceph-osd1:/dev/sdb ceph-osd2:/dev/sdb ceph-osd3:/dev/sdb` 在数据盘上面准备时也报错
   Running command: fdisk -l  File "/usr/lib/python2.7/distpackages/ceph_deploy/util/decorators.py", line 69, in newfunc 
   问题：未知
   解决办法：将osd节点的数据目录放在指定目录，不用整个数据盘
2. 最后部署后集群状况是health -ok，但是4osds，有3个osd up，一个osd down
   问题：down掉的节点磁盘有问题。
   解决办法：先卸载磁盘，重新格式化，挂载，重新激活osd节点

## 部署rgw节点出现的问题

1. 显示rgw进程在工作，但是使用：http://controller:7480 显示拒绝连接。并且新建S3账号，测试时未返回正确结果。

   问题：未知

   尝试方法：重新部署

   解决办法：重新部署后最开始将端口设置为80，发现可以创建s3账号，但是无法正确测试，显示创建bucket出错，查看rgw的log，发现端口被占用，无法打开，后面重新设置端口为7480问题解决，测试均正确。
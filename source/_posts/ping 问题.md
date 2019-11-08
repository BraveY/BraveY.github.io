---
title: ping 无法连接外网
date: 2019-03-15 13:51:47
categories: Linux
tags:
- Linux
- 运维
- 网络问题
copyright: true
---



## ping 无法连接外网

<!--more-->

### 问题

ping外网ping不通

```
yky@hw076:~/tmux> ping www.baidu.com
ping: unknown host www.baidu.com
yky@hw076:~/tmux> ping 8.8.8.8
connect: Network is unreachable
```

ping内网可以ping通

```
hw076:~ # ping 172.18.11.114
PING 172.18.11.114 (172.18.11.114) 56(84) bytes of data.
64 bytes from 172.18.11.114: icmp_seq=1 ttl=64 time=0.193 ms
64 bytes from 172.18.11.114: icmp_seq=2 ttl=64 time=0.216 ms
64 bytes from 172.18.11.114: icmp_seq=3 ttl=64 time=0.207 ms
64 bytes from 172.18.11.114: icmp_seq=4 ttl=64 time=0.200 ms
^C
--- 172.18.11.114 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 2999ms
rtt min/avg/max/mdev = 0.193/0.204/0.216/0.008 ms

```

ifconfig信息为：

```
hw076:~ # ifconfig 
eth0      Link encap:Ethernet  HWaddr 90:E2:BA:15:C9:C4  
          inet addr:172.18.11.76  Bcast:192.168.1.255  Mask:255.255.0.0
          inet6 addr: fe80::92e2:baff:fe15:c9c4/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:9725797 errors:0 dropped:506 overruns:0 frame:0
          TX packets:21023 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:598731249 (570.9 Mb)  TX bytes:2767270 (2.6 Mb)
          Memory:fb480000-fb500000 

lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:276 errors:0 dropped:0 overruns:0 frame:0
          TX packets:276 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:25088 (24.5 Kb)  TX bytes:25088 (24.5 Kb)
```

route显示路由信息如下：

```
hw076:/etc/netconfig.d # route
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
default         *               0.0.0.0         UG    0      0        0 eth0
loopback        *               255.0.0.0       U     0      0        0 lo
link-local      *               255.255.0.0     U     0      0        0 eth0
172.18.0.0      *               255.255.0.0     U     0      0        0 eth0
```

原因是route没有配置网关，gateway是空着的。

### 解决方法

通过查看其他可以正常访问的节点的路由信息，得知网关节点为：172.18.0.254。因此增加默认网关节点配置。

执行命令：

```
route add default  gw 172.18.0.254
```

再次查看路由信息：

```
hw076:~ # route
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
default         172.18.0.254    0.0.0.0         UG    0      0        0 eth0
loopback        *               255.0.0.0       U     0      0        0 lo
link-local      *               255.255.0.0     U     0      0        0 eth0
172.18.0.0      *               255.255.0.0     U     0      0        0 eth0
```

再次ping8.8.8.8显示正常，问题解决。
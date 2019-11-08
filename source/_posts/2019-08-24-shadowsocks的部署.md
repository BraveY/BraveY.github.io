---
title: shadowsocks的部署
date: 2019-08-24 12:59:32
categories: deploy
tags:
- shadowsocks
- 翻墙
- ipv6
copyright: true
---

# shadowsocks 的部署

部署shadowsocks主要有两个作用：

1. 可以翻墙 
2. 可以白嫖校园网的ipv6 

<!--more-->

## 1. 购买VPS服务器

目前使用vultr的VPS服务器，5刀一个月，使用的CENTOS发行版。貌似日本的服务器网速最快，而且支持IPV6。在部署服务器的时候就需要开启IPV6。

## 2. 服务器设置

### 安装

在/root/目录下创建文件夹

```shell
mkdir shadowsocks
cd shadowsocks/
```

安装python和pip工具以及git

```shell
yum install python-setuptools && easy_install pip
yum install git 
```

使用pip通过git安装shadowsocks

```
pip install git+https://github.com/shadowsocks/shadowsocks.git@master
ssserver
```

ssserver 命令用来查看是否安装成功

### 脚本

```
 vi shadowsocks.json
```

脚本内容为：

多端口账户脚本如下

```json
{
    "server":"::",
    "local_address": "127.0.0.1",
    "local_port":1080,
    "port_password":{
    "8388":"frankfurt123",
 	"2343":"password"
},
    "timeout":300,
    "method":"aes-256-cfb",
    "fast_open": false
}
```

其中server:"::" 用两个:: 冒号来表示可以同时使用ipv4和ipv6来访问服务器，如果就只有ipv4的地址的话，只能使用ipv4来翻墙。

通过脚本来启动shadowsocks

```shell
 ssserver -c /root/shadowsocks/shadowsocks.json -d start
 ssserver -c /root/shadowsocks/shadowsocks.json -d status
 ssserver -c /root/shadowsocks/shadowsocks.json -d stop
```

至此通过添加服务器配置就应该可以使用shadowsocks+switchomega客户端了，如果还不可以的话，多半是因为防火墙的问题。

### 防火墙设置

将8388添加到防火墙白名单。

```
firewall-cmd --zone=public --add-port=8388/tcp --permanent
firewall-cmd --reload
firewall-cmd --list-ports
```

## 客户端设置

我使用的是win10 可以从：<https://github.com/shadowsocks/shadowsocks-windows> 下载

MacOS：<https://github.com/shadowsocks/ShadowsocksX-NG/releases> 

在服务器设置界面依次添加IPV6的地址、端口、密码、加密方式就可以使用IPV6进行白嫖了。

## 参考

<https://github.com/shadowsocks/shadowsocks/tree/master> 
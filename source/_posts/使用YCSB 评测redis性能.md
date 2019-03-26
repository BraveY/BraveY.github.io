---
title: 使用YCSB 评测redis性能
date: 2019-03-12 17:11:44
categories:
tags:
- redis
- benchmark
---

# 使用YCSB 评测redis性能

YCSB是雅虎推出的可以评测许多主流数据库性能的基准测试，其中包括Redis。

<!--more-->

## 安装YCSB

1. 安装java和maven

   1. 机子已经有了java，所以只用安装maven Ubuntu安装命令为：

      `sudo apt-get install maven`

2. 安装YCSB 

   ````
   git clone http://github.com/brianfrankcooper/YCSB.git
   cd YCSB
   mvn -pl com.yahoo.ycsb:redis-binding -am clean package
   ````

   必须是gitclone的源码包才能执行mvn 命令。wget或者curl下来包是已经编译好了的无需执行mvn命令。

3. `mvn -pl com.yahoo.ycsb:redis-binding -am clean package` 报错：

   ````
   [INFO] Scanning for projects...
   [ERROR] [ERROR] Could not find the selected project in the reactor: com.yahoo.ycsb:redis-binding @ 
   [ERROR] Could not find the selected project in the reactor: com.yahoo.ycsb:redis-binding -> [Help 1]
   [ERROR] 
   [ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
   [ERROR] Re-run Maven using the -X switch to enable full debug logging.
   [ERROR] 
   [ERROR] For more information about the errors and possible solutions, please read the following articles:
   [ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MavenExecutionException
   ````

原因：此命令是在gitclone后未编译的时候使用的。而我之前是下载的编译好的tar.gz包，解压后是已经编译好了的。所以再次执行编译的命令时会报错。

## 使用YCSB

将redis-server启动后开始使用YCSB

### 设置数据库

需要先创建`usertable`的表，因为YCSB客户端默认是对`usertable` 进行操作。Redis将数据存储在内存中，不需要相关操作。

### 选择合适的DB interface

YCSB的操作是通过DB interface来实现的。最基本的DB interface是`com.yahoo.ycsb.BasicDB`，会将输出输出到`System.out`里。可以通过继承DB interface来自定义DB interface，也可以使用原有的DB interface。Redis不需要此步操作。

### 选择合适的负载

YCSB提供了6种负载，负载在worloads目录下。详情见<https://github.com/brianfrankcooper/YCSB/wiki/Core-Workloads>

1. **Workload A: Update heavy workload** 读写比例为： 50/50 混合负载 
2. **Workload A: Update heavy workload** 读写比例为：95/5  读为主的负载
3. **Workload C: Read only**  100% 的读  只读负载
4. **Workload D: Read latest workload**  读取最近的数据负载
5. **Workload E: Short ranges**  小范围的查询负载
6. **Workload F: Read-modify-write** 读修改写负载

自定义负载：参考<https://github.com/brianfrankcooper/YCSB/wiki/Implementing-New-Workloads>

可以通过修改参数文件或者新建java类来实现

需要注意的是YCSB的读写负载是针对哈希类型的数据而不是简单的字符串

### 指定需要的运行参数

主要是指定redis的ip ，端口，密码等。

命令如下：

```
./bin/ycsb load redis -s -P workloads/workloada -p "redis.host=127.0.0.1" -p "redis.port=6379" > outputLoad.txt
```

`-s` : **status**.十秒打印一次状态

### 加载负载

命令如下：

```
./bin/ycsb load redis -s -P workloads/workloada > outputLoad.txt
```

### 运行负载

命令如下：

```
./bin/ycsb run redis -s -P workloads/workloada > outputRun.txt
```

可以使用basic数据库来打印YCSB向数据库中写入的具体数据

```
bin/ycsb.sh load basic -P workloads/workloada
bin/ycsb.sh run basic -P workloads/workloada
```



## 参考

https://datawine.github.io/2018/12/11/YCSB%E9%A1%B9%E7%9B%AE%E5%AD%A6%E4%B9%A0/

https://github.com/brianfrankcooper/YCSB/tree/master/redis  


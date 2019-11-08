---
title: BigdataBench deploy
date: 2018-06-23 16:55:45
categories: deploy
tags: bigdatabench
copyright: true
---

# Bigdatabench 4.0 MPI版本 安装

<!--more-->

官网上面的指南BigDataBench User Manual有一些错误。

本机环境：

​	Centos 6.9

​	gcc (GCC) 4.8.2 20140120 (Red Hat 4.8.2-15)

​	g++ (GCC) 4.8.2 20140120 (Red Hat 4.8.2-15)

## mpi的安装

这部分网上资料很多，而Manual中有一点错误

1. 需要保证c 编译器 如gcc c++ 编译器 如：g++

2. 基础安装

   1. 从官网下载安装包解压

   - `wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz `  从官网下载安装包
   - `tar -zxvf mpich-3.2.1.tar.gz `  解压
   - `cd mpich-3.2.1`

   1. 配置安装目录   本机安装在mpich-install目录下

   - `./configure –prefix=/home/mpich-install 2>&1 | tee c.txt    ` 手册中&被错写为$了 `2>&1 | tee c.txt` 表示将输出的标准出错信息重定向到c.txt中。

   1. build

   - `make 2>&1 | tee m.txt `

   1. 安装

   - `make install 2>&1 | tee mi.txt `

   1. 将安装目录添加到PATH 环境变量中

   - `vim ~/.bashrc`
   - `export PATH=$PATH:/home/mpich-install/bin` 在最后一行添加
   - `source ~/.bashrc` 重启生效

3. 检查

   1. 检查路径
      - `which mpicc` 
      - `which mpic++`

4. 验证 

   在mpich的安装包目录下有提供例子程序运行

   1. `cd mpich-3.2.1/examples`
   2. `mpicc cpi.c -o cpi` 编译cpi.c程序求pi值
   3. `mpirun -n 4 ./cpi` 使用4个进程 注意`./`否则报错找不到文件

   如果是集群环境在每个节点将mpich安装在相同的路径然后编辑一个machine_file （里面是各个节点的host）然后`mpirun -f machine_file -n 3 ./cpi` 在集群上并行运行

## boost 安装

boost当前最新版本是：1.67 但是BigdataBench用的是1.43版本推荐安装这个旧版本

1. `wget https://sourceforge.net/projects/boost/files/boost/1.43.0/boost_1_43_0.tar.gz/download` 

2. 若下载下来的文件名为：downloads 则使用mv命令重命名在当前文件目录下:

   `mv downloads boost_1_43_0.tar.gz  `  

3. 解压`tar -zxvf boost_1_43_0.tar.gz`  之后`cd boost_1_43_0`

4. `sh bootstrap.sh`  执行这个命令运行脚本后会多出很多配置文件

5. 使用mpi,这一步骤很重要否则后续cmake时会提示找不到：boost_mpi

   1. 对低版本的boost 

      1. `which mpic++` 找mpich的目录

      2. `vim tools/build/v2/user-config.jam`

      3. 在最后添加： using mpi:后面是mpich的目录

         `#MPI config`

         `using mpi : /usr/lib64/mpich/bin/mpic++ ;`

   2. 对高版本的boost直接在boost_1_67_0目录下修改project-config.jam即可

6. ` ./bjam` 进行编译

7. `./bjam install` 这一步是必需的但在手册中没有表明。

## BigdataBench的配置

进入BigDataBench的安装根目录：

1. ` vim conf.properties` 添加$JAVA_HOME， $MPI_HOME ，$BigdataBench_HOMEMPI的路径
2. `sh prepar.sh` 

至此安装理论上已经成功。但仍然遇到了其他问题

## Perminsion denied问题

最开始的安装包是从windows下面考过去的结果生成cc的数据后无法运行执行脚本

![](BigdataBench-deploy\runcc.png)

原因是此时的run_connectedComponents已经不是可执行文件了（不是绿色的）需要`chmod a+x run_connectedComponents`来将文件的权限修改为可执行文件权限（修改后变为绿色）

后面wget下载后解压配置之后直接就是可执行文件！

## ldd 程序 动态链接库缺失

` [root@hw073 ConnectedComponent]# ldd run_connectedComponents`
`linux-vdso.so.1 =>  (0x00007ffdfc8d4000)`
`librt.so.1 => /lib64/librt.so.1 (0x0000003156e00000)`
`libpthread.so.0 => /lib64/libpthread.so.0 (0x0000003156a00000)`

`libboost_serialization-mt.so.1.43.0 => not found`
`libboost_filesystem-mt.so.1.43.0 => not found`
`libboost_system-mt.so.1.43.0 => not found`
`libstdc++.so.6 => /usr/lib64/libstdc++.so.6 (0x0000003162200000)`
`libm.so.6 => /lib64/libm.so.6 (0x0000003157200000)`
`libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x0000003161a00000)`
`libc.so.6 => /lib64/libc.so.6 (0x0000003156600000)`
`/lib64/ld-linux-x86-64.so.2 (0x0000003155e00000)`

最开始以为是没有指定LD_LIBRARY_PATH ，因为明明有这个文件的，后面使用find / -name 命令发现还是找不到，仔细一看ldd 的信息，发现上述文件都多了个-mt

解决办法： 在boost安装时的库。本机：`/usr/local/lib` 有着及其相似的3个文件`libboost_filesystem.so.1.43.0` 、`libboost_filesystem.so.1.43.0` ，`libboost_system.so.1.43.0` 均少了个-mt，因此将上述三个文件均拷贝一份命名为上述缺少的动态库文件。

`cd /usr/local/lib` #切换到对应的目录下

`cp libboost_system.so.1.43.0 libboost_system-mt.so.1.43.0 ` #拷贝为对应的文件名
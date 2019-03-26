---
title: make 2>&1 | tee log.txt 命令解析
date: 2018-06-23 17:21:19
categories: Linux
tags: 
- Linux
- command
---

# make 2>&1 | tee log.txt 命令解析

在安装mpich 的时候遇到了很多这个命令，此处学习下这个命令：`2>&1 | tee log.txt` 

<!--more-->

这个命令共有三个部分： `2>&1` `|`  `tee log.txt`

## 2>&1

shell中：最常使用的 FD (file descriptor) 大概有三个 

0表示标准输入Standard Input (STDIN)  

1表示标准输出Standard Output (STDOUT)  

 2表示标准错误输出 Standard Error Output (STDERR)  

'>' 默认为标准输出重定向 （类似于c++ 中的 >>？）

在标准情况下, 这些FD分别跟如下设备关联 

stdin(0): keyboard  键盘输入,并返回在前端   

stdout(1): monitor  正确返回值 输出到前端   

stderr(2): monitor 错误返回值 输出到前端  

1>&2  正确返回值传递给2输出通道 &2表示2输出通道   如果此处错写成 1>2, 就表示把1输出重定向到文件2中  2>&1 错误返回值传递给1输出通道, 同样&1表示1输出通道.  

## |管道

管道的作用是提供一个通道，将上一个程序的标准输出重定向到下一个程序作为下一个程序的标准输入。 

## tee log.txt

tee从标准输入中读取，并将读入的内容写到标准输出以及文件中。  此处将数据读入并写入到log.txt中

## 总结

这个命令将标准错误输出重定向到标准输出，然后再将标准输出重定向到log.txt文件中

常用于make 后面将log信息保存下来。
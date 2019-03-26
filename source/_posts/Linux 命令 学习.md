---
title: Linux命令学习之wc
date: 2018-07-9 21:12:11
categories: Linux
tags: 
- Linux
- command
---

# Linux 命令学习wc命令

## `wc`命令 

<!--more-->

1. 作用：Word Count 功能为统计指定文件中的字节数、字数、行数，并将统计结果显示输出。 
2. 格式：
   - `wc [option] filepath`
3. 参数
   - `-c` 统计字节数
   - `-l` 统计行数
   - `-m` 统计字符数 标志不能与 -c 标志一起使用。 
   - `-w` 统计字（单词word）数。一个字被定义为由空白、跳格或换行字符分隔的字符串 
   - `-L`  打印最长行的长度。 
   - `-help` 显示帮助信息 
   - `--version` 显示版本信息 
4. 参考网址：http://www.cnblogs.com/peida/archive/2012/12/18/2822758.html


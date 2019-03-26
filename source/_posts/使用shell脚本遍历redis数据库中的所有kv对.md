---
title: shell 脚本遍历redis数据库
date: 2019-03-14 20:00:46
categories: Linux
tags:
- shell
- redis
---

## 使用shell脚本遍历redis数据库中的所有kv对

记录下如何使用shell通过redis-cli 命令来操作redis数据库，因为直接在命令行中输入

`redis-cli command` 的话command必须是单个单词，不能像是`KEYS *` 这种.

<!--more-->

````
#!/bin/bash
filename='redis'`date +%Y-%m-%d_%H:%M`
work_path=$(dirname "$0") 
echo "实例化redis数据文件为:$work_path/$filename"
echo "keys *" | redis-cli > key_db.txt
echo "将所有key保存到:$work_path/key_db.txt"
for line in `cat key_db.txt`
do
        echo "key:$line " >>$work_path/$filename.txt
        echo "key-value:" >>$work_path/$filename.txt
        echo "hgetall $line" | redis-cli >>$work_path/$filename.txt
done
````

使用echo 来把命令输出到管道然后再传递给redis-cli。在循环里面也是使用echo来把字符串输入到文件中。
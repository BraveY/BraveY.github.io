---
title: Linux shell脚本计算运行时间
date: 2019-04-10 23:27:45
categories: Linux
tags: shell
---

# Linux shell脚本计算运行时间

这个功能经常用但是，总是现用现查，很麻烦。

<!--more-->

代码

```bash
# filename: msec_diff.sh

function timediff() {

# time format:date +"%s.%N", such as 1502758855.907197692
    start_time=$1
    end_time=$2
    
    start_s=${start_time%.*}
    start_nanos=${start_time#*.}
    end_s=${end_time%.*}
    end_nanos=${end_time#*.}
    
    # end_nanos > start_nanos? 
    # Another way, the time part may start with 0, which means
    # it will be regarded as oct format, use "10#" to ensure
    # calculateing with decimal
    if [ "$end_nanos" -lt "$start_nanos" ];then
        end_s=$(( 10#$end_s - 1 ))
        end_nanos=$(( 10#$end_nanos + 10**9 ))
    fi
    
# get timediff
    time=$(( 10#$end_s - 10#$start_s )).`printf "%03d\n" $(( (10#$end_nanos - 10#$start_nanos)/10**6 ))`
    
    echo $time
}

#start=$(date +"%s.%N")
# Now exec some command
#end=$(date +"%s.%N")
# here give the values
start=1502758855.907197692
end=1502758865.066894173

timediff $start $end
```

## 参考

<https://www.cnblogs.com/f-ck-need-u/p/7426987.html>
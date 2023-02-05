---
title: Redis源码阅读——dict
date: 2019-04-03 14:29:32
categories: 源码阅读
tags:
- redis
- dict
---

# Redis源码阅读——dict

继续Redis 的源码阅读，进入dict这一章节。知识点讲解，见redis设计与实现的读书笔记dict这一章。

<!--more-->

## dict的创建

还是和sds一样单独将dict模块给提取出来，参考博客是直接将server的main函数给修改了的。再阅读Makefile的时候发现了dict-benchmark 这个选项，`make dict-benchmark` 这个命令，可以编译出一个可执行文件dict-benchmark。 

所以想着应该可以单独再把dict-benchmark给提取出来。

需要拷贝的文件

```
dict.c ditc.h fmacros.h redisassert.h sdsaclloc.h sds.c sds.h siphash.c 
```

需要做的修改

zmalloc zfree zcalloc 需要修改成使用malloc ，free, calloc 其中zcalloc在修改成calloc的时候需要在调用的时候多传入一个参数1,作为第一个参数，因为zcalloc 和calloc 的接口不一样。

还需要在redisassert.h 中对_serverAssert函数做出定义： 参考debug.c 里面的定义。

```
void _serverAssert(char *estr, char *file, int line){
    printf("%s:%d'%s' is not true",file,line,estr);
    *((char*)-1) = 'x';
}
```

修改成使用libc 的内存分配器后会造成性能下降，zmalloc 要好于malloc 所以要想真实还原的话，看下怎么把zmalloc给移植过来。zmalloc用的是jemalloc.

编写简单的Makefile为：

```makefile
dict-benchmark: dict.c sds.c siphash.c
        $(CC) -g -o $@ $^
.PHONY :clean
clean :
        rm dict-benchmark 
```



## 计算索引

_dictKeyIndex 函数。

```c
/* Returns the index of a free slot that can be populated with
 * a hash entry for the given 'key'.
 * If the key already exists, -1 is returned
 * and the optional output parameter may be filled.
 *
 * Note that if we are in the process of rehashing the hash table, the
 * index is always returned in the context of the second (new) hash table. */
static int _dictKeyIndex(dict *d, const void *key, unsigned int hash, dictEntry **existing)
{
    unsigned int idx, table;
    dictEntry *he;
    if (existing) *existing = NULL;

    /* Expand the hash table if needed */
    if (_dictExpandIfNeeded(d) == DICT_ERR)
        return -1;
    for (table = 0; table <= 1; table++) {
        idx = hash & d->ht[table].sizemask;
        /* Search if this slot does not already contain the given key */
        he = d->ht[table].table[idx];
        while(he) {
            if (key==he->key || dictCompareKeys(d, key, he->key)) {
                if (existing) *existing = he;
                return -1;
            }
            he = he->next;
        }
        if (!dictIsRehashing(d)) break;
    }
    return idx;

```



## 参考

<https://blog.csdn.net/yangbodong22011/article/details/78467583>
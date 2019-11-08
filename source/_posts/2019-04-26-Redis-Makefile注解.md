---
title: Redis Makefile注解
date: 2019-04-26 15:57:19
categories: 源码阅读
tags:
- redis
- Makefile
copyright: true
---

# Redis Makefile注解

Redis的makefile是阅读源码的第一步，总共有292行，读起来也是头大，记录之。

<!--more-->

4.02版本源码为：

```makefile
# Redis Makefile
# Copyright (C) 2009 Salvatore Sanfilippo <antirez at gmail dot com>
# This file is released under the BSD license, see the COPYING file
#
# The Makefile composes the final FINAL_CFLAGS and FINAL_LDFLAGS using
# what is needed for Redis plus the standard CFLAGS and LDFLAGS passed.
# However when building the dependencies (Jemalloc, Lua, Hiredis, ...)
# CFLAGS and LDFLAGS are propagated to the dependencies, so to pass
# flags only to be used when compiling / linking Redis itself REDIS_CFLAGS
# and REDIS_LDFLAGS are used instead (this is the case of 'make gcov').
#
# Dependencies are stored in the Makefile.dep file. To rebuild this file
# Just use 'make dep', but this is only needed by developers.

release_hdr := $(shell sh -c './mkreleasehdr.sh')
# uname -s 获取操作系统的类型 Linux
uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')
#uname -m 获取机子的架构 x86_64
uname_M := $(shell sh -c 'uname -m 2>/dev/null || echo not')
# 优化选项
OPTIMIZATION?=-O2
# 依赖目标
DEPENDENCY_TARGETS=hiredis linenoise lua
NODEPS:=clean distclean

# Default settings
# 使用c99标准编译，-pedantic 保证代码规范满足ISO C和ISO C++标准
STD=-std=c99 -pedantic -DREDIS_STATIC=''
# 输出所有编译警告信息 ，Wno-missing-field-initializers 不输出missing-的警告信息
WARN=-Wall -W -Wno-missing-field-initializers
OPT=$(OPTIMIZATION)

#默认目录
PREFIX?=/usr/local
#安装的默认目录
INSTALL_BIN=$(PREFIX)/bin
INSTALL=install

# Default allocator defaults to Jemalloc if it's not an ARM
#内存分配器的指定 默认libc，linux系统而且架构不是armv61和71的时候则是jemalloc，
MALLOC=libc
ifneq ($(uname_M),armv6l)
ifneq ($(uname_M),armv7l)
ifeq ($(uname_S),Linux)
	MALLOC=jemalloc
endif
endif
endif

# To get ARM stack traces if Redis crashes we need a special C flag.
ifneq (,$(findstring armv,$(uname_M)))
        CFLAGS+=-funwind-tables
endif

# Backwards compatibility for selecting an allocator
#编译的时候指定内存分配器
ifeq ($(USE_TCMALLOC),yes)
	MALLOC=tcmalloc
endif

ifeq ($(USE_TCMALLOC_MINIMAL),yes)
	MALLOC=tcmalloc_minimal
endif

ifeq ($(USE_JEMALLOC),yes)
	MALLOC=jemalloc
endif

ifeq ($(USE_JEMALLOC),no)
	MALLOC=libc
endif

# Override default settings if possible
-include .make-settings
# 最终的编译选项CFLAGS是-c的选项，LDFLAGS是链接的选项
FINAL_CFLAGS=$(STD) $(WARN) $(OPT) $(DEBUG) $(CFLAGS) $(REDIS_CFLAGS)
FINAL_LDFLAGS=$(LDFLAGS) $(REDIS_LDFLAGS) $(DEBUG)
# m这个lib是libmath 也就是math的链接
FINAL_LIBS=-lm
# 调试信息
DEBUG=-g -ggdb
#根据操作系统继续指定编译选项
ifeq ($(uname_S),SunOS)
	# SunOS
        ifneq ($(@@),32bit)
		CFLAGS+= -m64
		LDFLAGS+= -m64
	endif
	DEBUG=-g
	DEBUG_FLAGS=-g
	export CFLAGS LDFLAGS DEBUG DEBUG_FLAGS
	INSTALL=cp -pf
	FINAL_CFLAGS+= -D__EXTENSIONS__ -D_XPG6
	FINAL_LIBS+= -ldl -lnsl -lsocket -lresolv -lpthread -lrt
else
ifeq ($(uname_S),Darwin)
	# Darwin
	FINAL_LIBS+= -ldl
else
ifeq ($(uname_S),AIX)
        # AIX
        FINAL_LDFLAGS+= -Wl,-bexpall
        FINAL_LIBS+=-ldl -pthread -lcrypt -lbsd
else
ifeq ($(uname_S),OpenBSD)
	# OpenBSD
	FINAL_LIBS+= -lpthread
else
ifeq ($(uname_S),FreeBSD)
	# FreeBSD
	FINAL_LIBS+= -lpthread
else
	# 特别是对Linux的指定
	# All the other OSes (notably Linux)
	# -rdynamic将链接器将所有符号添加到动态符号表
	FINAL_LDFLAGS+= -rdynamic
	#pthread库 用于多线程， dl是libdl 动态链接库
	FINAL_LIBS+=-ldl -pthread
endif
endif
endif
endif
endif
# Include paths to dependencies
# -I 指定头文件的目录
FINAL_CFLAGS+= -I../deps/hiredis -I../deps/linenoise -I../deps/lua/src

ifeq ($(MALLOC),tcmalloc)
	FINAL_CFLAGS+= -DUSE_TCMALLOC
	FINAL_LIBS+= -ltcmalloc
endif

ifeq ($(MALLOC),tcmalloc_minimal)
	FINAL_CFLAGS+= -DUSE_TCMALLOC
	FINAL_LIBS+= -ltcmalloc_minimal
endif
#使用jemalloc的话 链接 libjemalloc.a -I指定jemalloc的头文件目录
ifeq ($(MALLOC),jemalloc)
	DEPENDENCY_TARGETS+= jemalloc
	FINAL_CFLAGS+= -DUSE_JEMALLOC -I../deps/jemalloc/include
	FINAL_LIBS+= ../deps/jemalloc/lib/libjemalloc.a
endif
#redis 的gcc -c 选项
REDIS_CC=$(QUIET_CC)$(CC) $(FINAL_CFLAGS)
#redis的gcc 链接选项
REDIS_LD=$(QUIET_LINK)$(CC) $(FINAL_LDFLAGS)
#redis的安装选项
REDIS_INSTALL=$(QUIET_INSTALL)$(INSTALL)

CCCOLOR="\033[34m"
LINKCOLOR="\033[34;1m"
SRCCOLOR="\033[33m"
BINCOLOR="\033[37;1m"
MAKECOLOR="\033[32;1m"
ENDCOLOR="\033[0m"

ifndef V
QUIET_CC = @printf '    %b %b\n' $(CCCOLOR)CC$(ENDCOLOR) $(SRCCOLOR)$@$(ENDCOLOR) 1>&2;
QUIET_LINK = @printf '    %b %b\n' $(LINKCOLOR)LINK$(ENDCOLOR) $(BINCOLOR)$@$(ENDCOLOR) 1>&2;
QUIET_INSTALL = @printf '    %b %b\n' $(LINKCOLOR)INSTALL$(ENDCOLOR) $(BINCOLOR)$@$(ENDCOLOR) 1>&2;
endif

REDIS_SERVER_NAME=redis-server
REDIS_SENTINEL_NAME=redis-sentinel
# redis-server的需要使用的对象文件，也就是各个模块
REDIS_SERVER_OBJ=adlist.o quicklist.o ae.o anet.o dict.o server.o sds.o zmalloc.o lzf_c.o lzf_d.o pqsort.o zipmap.o sha1.o ziplist.o release.o networking.o util.o object.o db.o replication.o rdb.o t_string.o t_list.o t_set.o t_zset.o t_hash.o config.o aof.o pubsub.o multi.o debug.o sort.o intset.o syncio.o cluster.o crc16.o endianconv.o slowlog.o scripting.o bio.o rio.o rand.o memtest.o crc64.o bitops.o sentinel.o notify.o setproctitle.o blocked.o hyperloglog.o latency.o sparkline.o redis-check-rdb.o redis-check-aof.o geo.o lazyfree.o module.o evict.o expire.o geohash.o geohash_helper.o childinfo.o defrag.o siphash.o rax.o
REDIS_CLI_NAME=redis-cli
#redis-cli 需要使用的对象文件
REDIS_CLI_OBJ=anet.o adlist.o redis-cli.o zmalloc.o release.o anet.o ae.o crc64.o
REDIS_BENCHMARK_NAME=redis-benchmark
#redis-benchmark需要使用的对象文件
REDIS_BENCHMARK_OBJ=ae.o anet.o redis-benchmark.o adlist.o zmalloc.o redis-benchmark.o
REDIS_CHECK_RDB_NAME=redis-check-rdb
REDIS_CHECK_AOF_NAME=redis-check-aof
#所有需要需要构建的对象，第一条规则也就是默认规则，不指定规则的话，从第一个规则执行
all: $(REDIS_SERVER_NAME) $(REDIS_SENTINEL_NAME) $(REDIS_CLI_NAME) $(REDIS_BENCHMARK_NAME) $(REDIS_CHECK_RDB_NAME) $(REDIS_CHECK_AOF_NAME)
	@echo ""
	@echo "Hint: It's a good idea to run 'make test' ;)"
	@echo ""
#Makefil.dep 的生成
Makefile.dep:
	-$(REDIS_CC) -MM *.c > Makefile.dep 2> /dev/null || true

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS))))
-include Makefile.dep
endif

.PHONY: all
#先清除所有编译的输出然后，  将所有设置持久化
persist-settings: distclean
	echo STD=$(STD) >> .make-settings
	echo WARN=$(WARN) >> .make-settings
	echo OPT=$(OPT) >> .make-settings
	echo MALLOC=$(MALLOC) >> .make-settings
	echo CFLAGS=$(CFLAGS) >> .make-settings
	echo LDFLAGS=$(LDFLAGS) >> .make-settings
	echo REDIS_CFLAGS=$(REDIS_CFLAGS) >> .make-settings
	echo REDIS_LDFLAGS=$(REDIS_LDFLAGS) >> .make-settings
	echo PREV_FINAL_CFLAGS=$(FINAL_CFLAGS) >> .make-settings
	echo PREV_FINAL_LDFLAGS=$(FINAL_LDFLAGS) >> .make-settings
	-(cd ../deps && $(MAKE) $(DEPENDENCY_TARGETS))

.PHONY: persist-settings
 
# Prerequisites target
.make-prerequisites:
	@touch $@

# Clean everything, persist settings and build dependencies if anything changed
#当设置有变化的时候清除并重新持久化设置
ifneq ($(strip $(PREV_FINAL_CFLAGS)), $(strip $(FINAL_CFLAGS)))
.make-prerequisites: persist-settings
endif

ifneq ($(strip $(PREV_FINAL_LDFLAGS)), $(strip $(FINAL_LDFLAGS)))
.make-prerequisites: persist-settings
endif

# redis-server
#redis-server可执行程序的链接，需要链接的静态链接文件包括hiredi和lua,还有final_libs
$(REDIS_SERVER_NAME): $(REDIS_SERVER_OBJ)
	$(REDIS_LD) -o $@ $^ ../deps/hiredis/libhiredis.a ../deps/lua/src/liblua.a $(FINAL_LIBS)

# redis-sentinel
#redis-sentienl构建
$(REDIS_SENTINEL_NAME): $(REDIS_SERVER_NAME)
	$(REDIS_INSTALL) $(REDIS_SERVER_NAME) $(REDIS_SENTINEL_NAME)

# redis-check-rdb
#redis-check-rdb的构建
$(REDIS_CHECK_RDB_NAME): $(REDIS_SERVER_NAME)
	$(REDIS_INSTALL) $(REDIS_SERVER_NAME) $(REDIS_CHECK_RDB_NAME)

# redis-check-aof
#redis-check-aof的构建
$(REDIS_CHECK_AOF_NAME): $(REDIS_SERVER_NAME)
	$(REDIS_INSTALL) $(REDIS_SERVER_NAME) $(REDIS_CHECK_AOF_NAME)

# redis-cli
#redis-cli的链接
$(REDIS_CLI_NAME): $(REDIS_CLI_OBJ)
	$(REDIS_LD) -o $@ $^ ../deps/hiredis/libhiredis.a ../deps/linenoise/linenoise.o $(FINAL_LIBS)

# redis-benchmark
#redis-benchmark的链接
$(REDIS_BENCHMARK_NAME): $(REDIS_BENCHMARK_OBJ)
	$(REDIS_LD) -o $@ $^ ../deps/hiredis/libhiredis.a $(FINAL_LIBS)

dict-benchmark: dict.c zmalloc.c sds.c siphash.c
	$(REDIS_CC) $(FINAL_CFLAGS) $^ -D DICT_BENCHMARK_MAIN -o $@ $(FINAL_LIBS)

# Because the jemalloc.h header is generated as a part of the jemalloc build,
# building it should complete before building any other object. Instead of
# depending on a single artifact, build all dependencies first.
#将所有点c文件编译成.o文件 自动完成file.c 到file.o的对应
%.o: %.c .make-prerequisites
	$(REDIS_CC) -c $<

clean:
	rm -rf $(REDIS_SERVER_NAME) $(REDIS_SENTINEL_NAME) $(REDIS_CLI_NAME) $(REDIS_BENCHMARK_NAME) $(REDIS_CHECK_RDB_NAME) $(REDIS_CHECK_AOF_NAME) *.o *.gcda *.gcno *.gcov redis.info lcov-html Makefile.dep dict-benchmark

.PHONY: clean

distclean: clean
	-(cd ../deps && $(MAKE) distclean)
	-(rm -f .make-*)

.PHONY: distclean

test: $(REDIS_SERVER_NAME) $(REDIS_CHECK_AOF_NAME)
	@(cd ..; ./runtest)

test-sentinel: $(REDIS_SENTINEL_NAME)
	@(cd ..; ./runtest-sentinel)

check: test

lcov:
	$(MAKE) gcov
	@(set -e; cd ..; ./runtest --clients 1)
	@geninfo -o redis.info .
	@genhtml --legend -o lcov-html redis.info

test-sds: sds.c sds.h
	$(REDIS_CC) sds.c zmalloc.c -DSDS_TEST_MAIN $(FINAL_LIBS) -o /tmp/sds_test
	/tmp/sds_test

.PHONY: lcov

bench: $(REDIS_BENCHMARK_NAME)
	./$(REDIS_BENCHMARK_NAME)

32bit:
	@echo ""
	@echo "WARNING: if it fails under Linux you probably need to install libc6-dev-i386"
	@echo ""
	$(MAKE) CFLAGS="-m32" LDFLAGS="-m32"

gcov:
	$(MAKE) REDIS_CFLAGS="-fprofile-arcs -ftest-coverage -DCOVERAGE_TEST" REDIS_LDFLAGS="-fprofile-arcs -ftest-coverage"

noopt:
	$(MAKE) OPTIMIZATION="-O0"

valgrind:
	$(MAKE) OPTIMIZATION="-O0" MALLOC="libc"

helgrind:
	$(MAKE) OPTIMIZATION="-O0" MALLOC="libc" CFLAGS="-D__ATOMIC_VAR_FORCE_SYNC_MACROS"

src/help.h:
	@../utils/generate-command-help.rb > help.h
#将构建完成的可执行程序安装到指定的目录，-p选项自行创建多层目录
install: all
	@mkdir -p $(INSTALL_BIN)
	$(REDIS_INSTALL) $(REDIS_SERVER_NAME) $(INSTALL_BIN)
	$(REDIS_INSTALL) $(REDIS_BENCHMARK_NAME) $(INSTALL_BIN)
	$(REDIS_INSTALL) $(REDIS_CLI_NAME) $(INSTALL_BIN)
	$(REDIS_INSTALL) $(REDIS_CHECK_RDB_NAME) $(INSTALL_BIN)
	$(REDIS_INSTALL) $(REDIS_CHECK_AOF_NAME) $(INSTALL_BIN)
	@ln -sf $(REDIS_SERVER_NAME) $(INSTALL_BIN)/$(REDIS_SENTINEL_NAME)

```

`uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not') ` 这一句-c让后面的字符串命令当成一个完成的命令来执行，从而避免向文件中写入东西的时候权限不够的问题。就算加上sudo也不行，因为里面的命令有>,echo等很多个文件，所以只能用-c来当成一个整体来执行。[参考](https://blog.csdn.net/bobchill/article/details/84647575)

## Makefile思路

总结一下Redis Makefile的思路：

1. 在默认规则也就是第一条规则之前，通过变量设置好编译的相关选项：LDFLAGS，相应的对应关系REDIS_SERVER_OBJ，将规则的target用变量表示好（方便all规则里面用作前置条件），比如REDIS_SERVER_NAME。
2. 在第一条默认规则 all规则里面指定需要构建的东西
3. 在第一规则后面先完成链接，再完成编译的规则
4. 其他功能性规则如clean和distclean

也就是从上到下的结构是总-分。显示整个项目 ，然后是各个模块如redis-server，redis-cli的链接，然后是从源文件到obj文件的编译。

## 参考

<https://blog.csdn.net/bobchill/article/details/84647575>
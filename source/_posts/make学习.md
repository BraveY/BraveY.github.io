---
title: Make学习
date: 2018-10-09 22:51:57
categories: 编程语言
tags:
- make
---

# make学习

开始阅读redis源码，都说redis很简单，源码不多。但是源码包下载下来后却发现不知道从何处入手，有那么多文件和源码。后面查找资料才发现阅读源码的第一步就是阅读Makefile，项目如何构建和源码间的关联都写在了Makefile文件中。之前没有接触过Makefile，记录下Make的学习。

<!--more-->

## makefile的格式

1. 概述

   makefile 文件由一系列rules组成 rules的格式为：

```
<target> : <prerequisites> 
[tab]  <commands>
```

​	"目标"是必需的，不可省略；"前置条件"和"命令"都是可选的，但是两者之中必须至少存在一个。 

​	每条规则就明确两件事：构建目标的前置条件是什么，以及如何构建。 

2. target

   一个目标（target）就构成一条规则。目标通常是文件名，指明Make命令所要构建的对象，比如上文的 a.txt 目标可以是一个文件名，也可以是多个文件名，之间用空格分隔。（make的时候指定文件名从而对该文件进行构建build）

   除了文件名，目标还可以是某个操作的名字，这称为"伪目标"（phony target）。伪目标不生成文件，只执行命令。

   比如：

   ```
   clean:
         rm *.o
   ```

   此时执行`make clean` 命令则会进行`rm *.o` 的操作。

   但是当存在clean这个文件时，那么这个命令不会执行。因为Make发现clean文件已经存在，就认为没有必要重新构建了，就不会执行指定的rm命令。

   为了避免这种情况，可以明确声明clean是"伪目标"，写法如下。

   ```
   .PHONY: clean
   clean:
           rm *.o temp
   ```

   如果Make命令运行时没有指定目标，默认会执行Makefile文件的第一个目标。 

3. prerequisites

   前置条件通常是一组文件名，之间用空格分隔。它指定了"目标"是否重新构建的判断标准：只要有一个前置文件不存在，或者有过更新（前置文件的last-modification时间戳比目标的时间戳新），"目标"就需要重新构建。 

   没有前置条件，就意味着它跟其他文件都无关，只要这个target文件还不存在 就需要执行命令构建

   如果需要生成多个文件，往往采用下面的写法。 

   `source: file1 file2 file3`  

   无需加上命令，当三个文件不存在时，执行`make source`就会生成这三个文件。

4. commands

   命令（commands）表示如何更新目标文件，由一行或多行的Shell命令组成。它是构建"目标"的具体指令，它的运行结果通常就是生成目标文件。 

   每行命令之前必须有一个tab键 

   需要注意的是，每行命令在一个单独的shell中执行。这些Shell之间没有继承关系。

   ```
   var-lost:
       export foo=bar
       echo "foo=[$$foo]"
   ```

   上面代码执行后（`make var-lost`），取不到foo的值。因为两行命令在两个不同的进程执行。 

   解决办法：

    1. 命令写在同1行

    2. 换行符前加反斜杠转义

       ```
       var-kept:
           export foo=bar; \
           echo "foo=[$$foo]"
       ```

   	3. 加上`.ONESHELL:`命令 

       ```
       .ONESHELL:
       var-kept:
           export foo=bar; 
           echo "foo=[$$foo]"
       ```

## makefile的语法

1. 注释

   井号（#）在Makefile中表示注释。 

2. 回声（echoing）

   正常情况下，make会打印每条命令，然后再执行，这就叫做回声（echoing）。

   在命令的前面加上@，就可以关闭回声。 

   由于在构建过程中，需要了解当前在执行哪条命令，所以通常只在注释和纯显示的echo命令前面加上@。 

3. 通配符

   由于在构建过程中，需要了解当前在执行哪条命令，所以通常只在注释和纯显示的echo命令前面加上@。 

4. 模式匹配

   Make命令允许对文件名，进行类似正则运算的匹配，主要用到的匹配符是%。比如，假定当前目录下有 f1.c 和 f2.c 两个源码文件，需要将它们编译为对应的对象文件。 

   ```
   %.o: %.c
   ```

   等同于

   ```
   f1.o: f1.c
   f2.o: f2.c
   ```

   使用匹配符%，可以将大量同类型的文件，只用一条规则就完成构建。 

5. 变量和赋值符

   Makefile 允许使用等号自定义变量。 

   ```
   txt = Hello World
   test:
       @echo $(txt)
   ```

   上面代码中，变量 txt 等于 Hello World。调用时，变量需要放在 $( ) 之中 

   调用Shell变量，需要在美元符号前，再加一个美元符号，这是因为Make命令会对美元符号转义。 

6. 内置变量

   Make命令提供一系列内置变量，比如，$(CC) 指向当前使用的编译器，$(MAKE) 指向当前使用的Make工具。这主要是为了跨平台的兼容性 gmake、cmake、dmake等等。

   $(AR) ：函数库打包程序,将对应的gcc编译出来的obj文件打包成静态链接库程序。

   ar可以集合许多文件，成为单一的备存文件。在备存文件中，所有成员文件皆保有原来的属性与权限。

7. 自动变量

   1. $@指代当前目标，就是Make命令当前构建的那个目标  target

   2. $<指代第一个前置条件。比如，规则为 t: p1 p2，那么$< 就指代p1 

   3. $？指代比目标更新的所有前置条件，之间以空格分隔。比如，规则为 t: p1 p2，其中 p2 的时间戳比 t 新，$?就指代p2。 

   4. $^指代所有前置条件，之间以空格分隔。比如，规则为 t: p1 p2，那么 $^ 就指代 p1 p2 。 

   5. $*指代匹配符 % 匹配的部分， 比如% 匹配 f1.txt 中的f1 ，$* 就表示 f1。 

   6. $(@D) 和 $(@F)$(@D) 和 $(@F) 分别指向 $@ 的目录名和文件名。比如，$@是 src/input.c，那么$(@D) 的值为 src ，$(@F) 的值为 input.c。 

   7. $(<D) 和 $(<F)

      $(<D) 和 $(<F) 分别指向 $< 的目录名和文件名。

8. 其他

   1. `.DEFAULT：`表示找不到匹配规则时，就执行该recipe。  

      ```
      default:all
      .DEFAULT:
      	commands
      ```

      这里当执行`make default` 时会转到`make all` 因为default：all 这个target没有隐式规则。所以最后会执行commands。

   2. 忽略命令的出错，可以在Makefile的命令行前加一个减号"-"(在Tab键之后)，标记为不管命令出不出错都认为是成功的。如：     

      ```
      clean:        
      	-(rm -f *.o )
      ```

   3. `include filename` 将filename中的内容导入，如果找不到会停止make， `-include filename` 则不会停止make。 

## 几种等号

= 是最基本的赋值
:= 是覆盖之前的值
?= 是如果没有被赋值过就赋予等号后面的值
+= 是添加等号后面的值

=与:= 的区别

 =：make会将整个makefile展开后，再决定变量的值。也就是说，变量的值将会是整个makefile中最后被指定的值。例子为：

```makefile
     x = foo
     y = $(x) bar
     x = xyz
```

y的值将会是 xyz bar ，而不是 foo bar 。因为展开后最终变成的是xyz

:=表示变量的值决定于它在makefile中的位置，而不是整个makefile展开后的最终值。

```makefile
 x := foo
 y := $(x) bar
 x := xyz
```

y的值将会是 foo bar ，而不是 xyz bar 了。

## 参考资料：

1. http://www.ruanyifeng.com/blog/2015/02/make.html
2. https://gist.github.com/isaacs/62a2d1825d04437c6f08 makefile文件教程
3. https://www.gnu.org/software/make/manual/make.html GNUmake手册
4. <https://blog.csdn.net/shouso888/article/details/7226030> 等号解释


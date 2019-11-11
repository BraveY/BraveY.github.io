---
title: OJ输入输出
date: 2019-09-08 22:16:11
categories: 题解
tags:
- OJ
- 编程
copyright: true
---

# OJ输入输出

算法课要求打UOJ，实际操作后发现与leetcode，牛客这些只用写解决类不一样，OJ要求自己编写输入输出。所以对于输入输出还是很头痛，在此总结下。

<!--more-->

## C++输入输出

输入输出不是对文件进行操作的，可以理解成是在命令行中进行输入与输出。所以主要使用标准输入流cin进行数据的输入，标准输出流cout进行输出。因为有多组测试样例，所以一般需要放在while循环中来读取数据并进行操作。总思路是输入一组输入对应的输出一组输出，边输入边输出。

需要注意的是cin 会自动跳过空格、tab、换行符等不可见的符号，所以可以在同一行中输入a,b两个值，而不用自己去分割空格。

### 只有一组输入输出 	

直接从键盘获取一组输入，随后输出，以计算a+b为例。

```c++
#include < iostream >   
using namespace std; 
int main() 
{
     int a,b; 
     cin >> a >> b;
     cout << a+b << endl; 
     return 0; 
}
```

## **有多组测试数据，直到读至输入文件结尾为止** 

有多组测试数据，需要在while循环中读取数据并进行处理。当输入

````c++
#include < iostream >    
using namespace std;
int main()
{
       int a,b;
       while(cin >> a >> b)
            cout << a+b << endl;
       return 0;
}
````

## **在开始的时候输入一个N，接下来是N组数据** 

在while循环中进行数据读入，需要注意的是如果后面需要用到n这个参数，需要使用临时变量来存储n，否则n在循环后会变成0.

```c++
#include <iostream>
using namespace std;
int main() {
    int a, b, n;
    cin >> n;
    while (n--) {
        cin>>a>>b;
        cout << a + b << endl;
    }
    return 0;
}
```

## 未知输入数据量，但以某个特殊输入为结束标志

当a或者b为0的时候结束输入，否则读入一组a，b并输出二者之和。

```c++
#include<iostream>
using namespace std;
int main()
{
    int a ,b;
    while(cin>>a>>b&&(a||b)){
        cout<<a+b<<endl;
    }
    return 0;
}
```

## 重定向输入

将输入从控制台重定向到文件，从文件进行输入。

```c++
#include<iostream>  
#include<cstdio>  
using namespace std;  
int main()  
{  
    freopen("input.txt","r",stdin);  //输入将被重定向到文件
    int a,b;  
    cin>>a>>b;  
    cout<<a+b<<endl;  
    return 0;  
} 
```

## 字符串输入

使用` cin.getline()`函数，其原型为：

```c++
istream& getline(char line[], int size, char endchar = '\n');
char line[]： 就是一个字符数组，用户输入的内容将存入在该数组内。
int size : 最多接受几个字符，用户超过size的输入都将不被接受。
char endchar :当用户输入endchar指定的字符时，自动结束，默认是回车符。
```

所以输入指定数目的字符串可以写成：

```c++
#include<iostream>
using namespace std;
int main()
{
    char buf[ 255 ];
    while(cin.getline( buf, 255 ));

}
```

也可以使用string类型来进行输入，如下程序循环输入pair组字符串，每组字符串有两个字符串用空格分开。

```c++
#include<iostream>
#include<string>
using namespace std;
int main(int argc, char const *argv[])
{
	int pair;
    string str1, str2;
	while(cin>>pair){
		while(pair--){
			cin>>str1;
			cin>>str2;
			cout<<str1<<str2<<endl;
		}
	}
	return 0;
}
```

上面的输入样式为：

```
2
ABCD AEFC
SCFEZ BNI
3
ABCD AEFC
SCFEZ BNI
ABCD XVC
```

即第一次输入2组字符串，第一组字符串为：`ABCD 与 AEFC`这两个字符串，cin会跳过空格即自动把空格前的ABCD这个字符串作为str1的输入而把空格后面的`AEFC`作为str2的输入。第二组字符串为：`SCFEZ BNI` ,与前面同理不赘述。

第二次输入为3组字符串，与第一次同理。

## 参考

<https://blog.csdn.net/qiao1245/article/details/53020326> 

<https://www.cnblogs.com/emerson027/articles/9319135.html> 
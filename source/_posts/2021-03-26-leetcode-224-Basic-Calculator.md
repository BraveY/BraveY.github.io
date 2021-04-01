---
title: leetcode 224 Basic Calculator
date: 2021-03-26 16:14:10
categories: 题解
tags:
- 栈
copyright: true
---

## 题意

实现带括号和加减号的计算器。[题目链接](https://leetcode.com/problems/basic-calculator/)

Given a string `s` representing an expression, implement a basic calculator to evaluate it.

**Example 1:**

```
Input: s = "1 + 1"
Output: 2
```

**Example 2:**

```
Input: s = " 2-1 + 2 "
Output: 3
```

**Example 3:**

```
Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23
```

**Constraints:**

- `1 <= s.length <= 3 * 105`
- `s` consists of digits, `'+'`, `'-'`, `'('`, `')'`, and `' '`.
- `s` represents a valid expression.

## 方法1 栈

### 思路

计算的时候存在括号匹配的问题，所以可以使用栈来进行匹配的运算。这里入栈的不是左括号，而是左括号之前的运算结果。

对于输入字符串来说，每次遍历的字符总共只有5种情况：

1. 字符为数字 说明还在读取一个数字，使用字符串转为数字的方法：`num = num*10 + cur`
2. 字符为加号或者减号，说明之前的数字已经读取完成需要将当前数字乘以对应的符号，加入到当前的和result种。并重新开始下一个数字的读取（sign为1或者-1）。
3. 字符为左括号：说明前面的部分已经计算完成，所以将结果和符号（括号里面计算的结果的符号）压入栈。
4. 字符为右括号：说明括号中的计算完成，进行出栈的操作。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

```cc
class Solution {
public:
    int calculate(string s) {
        int result = 0;
        int number = 0;
        int sign = 1;
        stack<int> stackCal;
        for(int i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (isdigit(c)) {
                number = number * 10 + (int) (c - '0');
            }else if (c == '+') {
                result += sign * number;
                number = 0;
                sign = 1;
            }else if (c == '-') {
                result += sign * number;
                number = 0;
                sign = -1;
            }else if (c == '(') {
                //result += sign * number; // 左括号前面的只能时符号而不是数字，所以左括号之前已经做了运算。
                stackCal.push(result);
                stackCal.push(sign);
                result = 0;
                sign = 1;
            }else if (c == ')') {
                result += sign * number; // 右括号前面的只能是数字而不是符号，所以右括号之前需要做运算
                result *= stackCal.top(); //栈的操作符是给括号里面的
                stackCal.pop();
                result += stackCal.top();
                stackCal.pop();
                number = 0;
                sign = 1;
            }
        }
        if (number != 0) result += sign * number; // 数字还没加到result上的情况
        return result;
    }
};
```

## 总结

1. 加与减统一为进行加的操作，只是减的时候是加一个负数。因此有个符号与数字分开的方法。
2. 考虑好完善的情况再进行编码。自己之前写的程序只能处理`1+1+1`这样的情况，没考虑`1++--+1`这样符号连续的情况。

## 参考

[leetcode讨论区](https://leetcode.com/problems/basic-calculator/discuss/62361/Iterative-Java-solution-with-stack)
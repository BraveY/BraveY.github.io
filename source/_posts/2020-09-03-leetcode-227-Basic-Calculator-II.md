---
title: leetcode 227 Basic Calculator II
date: 2020-09-03 12:49:09
categories: 题解
tags:
- String
- 栈
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/basic-calculator-ii/)对字符串表达式进行运算。

Implement a basic calculator to evaluate a simple expression string.

The expression string contains only **non-negative** integers, `+`, `-`, `*`, `/` operators and empty spaces ``. The integer division should truncate toward zero.

**Example 1:**

```
Input: "3+2*2"
Output: 7
```

**Example 2:**

```
Input: " 3/2 "
Output: 1
```

**Example 3:**

```
Input: " 3+5 / 2 "
Output: 5
```

**Note:**

- You may assume that the given expression is always valid.
- **Do not** use the `eval` built-in library function.

## 方法 栈

### 思路

表达式的组成分为操作数与运算符两个部分，在首个操作数前加上一个‘+’，则每个操作数都可以分配到左边的一个运算符从而绑定成一个操作对，比如`"3+5/2"` 可以理解成`"+3+5/2"`。之后按顺序读取操作对，如果是加减操作对则将对应的操作数变为对应的正数或者负数压入栈，因为加减的运算级别小于乘除需要最后计算。如果是乘除操作对则立即进行运算，栈顶的操作数为左操作数，当前操作对作为右操作数，然后将乘除运算的结果更新到栈顶。最后再将栈中的所有操作数累加就得到结果了。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$,

### 代码

花花的代码

```cc
/*
Runtime: 20 ms, faster than 92.76% of C++ online submissions for Basic Calculator II.
Memory Usage: 8.9 MB, less than 32.05% of C++ online submissions for Basic Calculator II.
 */
class Solution {
  public:
	int calculate(string s) {
		vector<int> nums;
		char op = '+';
		int cur = 0;
		int pos = 0;
		while (pos < s.size()) {
			if (s[pos] == ' ') {
				++pos;
				continue;
			}
			while (isdigit(s[pos]) && pos < s.size())
				cur = cur * 10 + (s[pos++] - '0'); // left to right way of Str2int
			if (op == '+' || op == '-') {
				nums.push_back(cur * (op == '+' ? 1 : -1));
			} else if (op == '*') {
				nums.back() *= cur;
			} else if (op == '/') {
				nums.back() /= cur;
			}
			cur = 0;
			op = s[pos++];
		}
		return accumulate(begin(nums), end(nums), 0);
	}
};
```

需要注意的是从左到右直接实现字符串到数字的实现，先前读取到的数字一直乘以10就可以移位了，不一定非要从后向前来进行。

## 方法2 数组

### 思路

自己的写法，写的很冗余不具有参考性，所以只简单叙述了。读取到的操作数和操作符都放在数组里面，操作符都是负数，然后先执行乘除操作，将计算过的数目也置为负数，最后再进行加减操作。

### 代码

```cc
/*
Runtime: 40 ms, faster than 35.74% of C++ online submissions for Basic Calculator II.
Memory Usage: 12 MB, less than 11.05% of C++ online submissions for Basic Calculator II.
 */
class Solution1 {
  public:
	int calculate(string s) {
		int len = s.length();
		vector<int> first;
		vector<long long > math;
		int strLen = 0;
		int mathLen = 0;
		string opStr = "";
		for (int i = 0; i < len; ++i) {
			if (s[i] == ' ') continue;
			if (s[i] == '/' || s[i] == '*') {
				mathLen += 2;
				first.push_back(mathLen - 1 );
				math.push_back(str2int(opStr));
				if (s[i] == '/') math.push_back(-2);
				else math.push_back(-1);
				opStr = "";
			} else if (s[i] == '+' || s[i] == '-') {
				mathLen += 2;
				math.push_back(str2int(opStr));
				if (s[i] == '+') math.push_back(-3);
				else math.push_back(-4);
				opStr = "";
			} else {
				opStr += s[i];
			}
		}
		math.push_back(str2int(opStr));

		for (int i = 0; i < first.size(); ++i) {
			int j = first[i] - 1;
			if (math[first[i]] == -1)  {
				while (math[j] < 0) --j;
				math[first[i]] = math[j] * math[first[i] + 1];
			} else {
				while (math[j] < 0) --j;
				math[first[i]] = math[j] / math[first[i] + 1];
			}
			math[j] = -5;
			math[first[i] + 1] = -5;
		}

		int ans = 0;
		for (int i = 0; i < math.size(); ++i) {
			if (math[i] >= 0) {
				ans += math[i];
			} else if (math[i] == -4) {
				while (math[i] < 0 ) ++i;
				ans -= math[i];
			}
		}
		return ans;
	}

  private:
	long long  str2int(string str) {
		int len = str.size();
		long long  digit = 1;
		long long ans = 0;
		while (len--) {
			ans += (str[len] - '0') * digit;
			digit *= 10;
		}
		return ans;
	}
};
```



## 总结

1. 栈适合对有顺序匹配的问题
2. str2int的从左至右写法
3. 自己写的时候出现了很多边界情况，然后一直对着样例修修补补才写出来的，这样修补很多次的一般都是方法没想好。

## 参考

[花花leetcode](https://zxi.mytechroad.com/blog/stack/leetcode-227-basic-calculator-ii/)
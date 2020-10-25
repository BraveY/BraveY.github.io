---
title: leetcode 338 Counting Bits
date: 2020-10-25 15:15:11
categories: 题解
tags:
- 动态规划
- 位操作
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/counting-bits/)

Given a non negative integer number **num**. For every numbers **i** in the range **0 ≤ i ≤ num** calculate the number of 1's in their binary representation and return them as an array.

**Example 1:**

```
Input: 2
Output: [0,1,1]
```

**Example 2:**

```
Input: 5
Output: [0,1,1,2,1,2]
```

**Follow up:**

- It is very easy to come up with a solution with run time **O(n\*sizeof(integer))**. But can you do it in linear time **O(n)** /possibly in a single pass?
- Space complexity should be **O(n)**.
- Can you do it like a boss? Do it without using any builtin function like **__builtin_popcount** in c++ or in any other language.

## 方法1 动态规划

### 思路

题目要求$O(N)$的时间复杂度，说明需要用到之前的结果，也就是动态规划的方法。但是动态规划的核心状态转移方程上比较难想。将当前数i与前一个数(i-1)进行&的操作，也就是将最右边的一个1转为0，因为少了最右边的一个1，所以其结果是少了最右边的一个1的数字的结果加1，也就是`ans[i] = ans[i & (i - 1)] + 1`;

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 8 ms, faster than 70.04% of C++ online submissions for Counting Bits.
Memory Usage: 8.3 MB, less than 35.13% of C++ online submissions for Counting Bits.
 */
class Solution {
  public:
	vector<int> countBits(int num) {
		vector<int> ans(num + 1, 0);
		for (int i = 1; i <= num; ++i) {
			ans[i] = ans[i & (i - 1)] + 1;
		}
		return ans;
	}

  private:
};
```

## 总结

1. 涉及到位操作的，一般就是与，或两种逻辑

## 参考

https://leetcode.com/problems/counting-bits/discuss/79527/Four-lines-C%2B%2B-time-O(n)-space-O(n)
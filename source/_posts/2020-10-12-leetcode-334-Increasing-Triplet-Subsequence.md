---
title: leetcode 334 Increasing Triplet Subsequence
date: 2020-10-12 11:01:53
categories: 题解
tags: 
- 数组
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/increasing-triplet-subsequence/) 判断数组是否有长度为3的递增子序列。

Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.

Formally the function should:

> Return true if there exists *i, j, k*
> such that *arr[i]* < *arr[j]* < *arr[k]* given 0 ≤ *i* < *j* < *k* ≤ *n*-1 else return false.

**Note:** Your algorithm should run in O(*n*) time complexity and O(*1*) space complexity.

**Example 1:**

```
Input: [1,2,3,4,5]
Output: true
```

**Example 2:**

```
Input: [5,4,3,2,1]
Output: false
```

## 方法1

### 思路

使用两个变量c1，与c2，其中c1记录当前为止的最小值，c2则记录除了当前为止的第二小值。如果出现了第三小的值，则找到了符合条件的子序列。 如果直到遍历完成也没有出现第三小的值，则找不到符合条件的序列。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(1)$

### 代码

```cc
/*
Runtime: 12 ms, faster than 51.92% of C++ online submissions for Increasing Triplet Subsequence.
Memory Usage: 10.3 MB, less than 100.00% of C++ online submissions for Increasing Triplet Subsequence.
 */
class Solution {
  public:
	bool increasingTriplet(vector<int>& nums) {
		int n = nums.size();
		int c1 = INT_MAX, c2 = INT_MAX;
		for (int i = 0; i < n; ++i) {
			if (nums[i] <= c1) c1 = nums[i];
			else if (nums[i] <= c2) c2 = nums[i];
			else {
				return true;
			}
		}
		return false;
	}

  private:
};
```

## 总结

1. 寻找第二小的写法，设置两个状态变量和更新的先后顺序。
2. c1，和c2二者都在动态更新，但是优先更新c1。
3. 核心是寻找$c1<c2<nums[x]$这样的顺序。

## 参考

[disscuss](https://leetcode.com/problems/increasing-triplet-subsequence/discuss/78993/Clean-and-short-with-comments-C%2B%2B)
---
title: leetcode 41 First Missing Positive
date: 2020-09-19 20:31:47
categories: 题解
tags:
- 数组
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/first-missing-positive/submissions/) 寻找数组中没出现的最小正数。

Given an unsorted integer array, find the smallest missing positive integer.

**Example 1:**

```
Input: [1,2,0]
Output: 3
```

**Example 2:**

```
Input: [3,4,-1,1]
Output: 2
```

**Example 3:**

```
Input: [7,8,9,11,12]
Output: 1
```

**Follow up:**

Your algorithm should run in *O*(*n*) time and uses constant extra space.

## 方法1 交换

### 思路

题目要求的是最小的未出现正数，所以肯定有个连续区间的问题，又要求在常数空间复杂度下进行$O(1)$的查询去判断某个数是否在数组中，所以想到利用原始数组来重新存放数字，并且根据索引来直接存储数字，将数字与索引绑定来实现哈希函数，从而可以实现$O(1)$的查询。为了与索引绑定需要进行交换，将每个数字放到对应的位置上。

自己最初的想法是找到最小正数值，然后最小正数值放到第一位，之后依次顺序填充。之后发现需要排序，以及解决重复值的问题，被卡在这了。

正确的做法是只考虑在数组大小内的正数，将数组大小内的正数放到对应的索引位置上。超过范围的正数与负数都不考虑，这样就可以将原始数组重新整理成范围内的正数都在对应位置上的新数组。之后再次遍历这个数组，如果当前值与对应的索引不一致，则当前值就是缺少的最小正数，如果所有的都对应则缺失值就是数组大小的下一个正数。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(1)$

### 代码

```cc
/*
Runtime: 4 ms, faster than 83.55% of C++ online submissions for First Missing Positive.
Memory Usage: 9.8 MB, less than 59.20% of C++ online submissions for First Missing Positive.
 */
class Solution {
  public:
	int firstMissingPositive(vector<int>& nums) {
		int n = nums.size();
		for (int i = 0; i < n; ++i) {
			while (nums[i] <= n && nums[i] > 0 && nums[nums[i] - 1] != nums[i]) {
				swap(nums[i], nums[nums[i] - 1]);
			}
		}
		for (int i = 0; i < n; ++i) {
			if (nums[i] != i + 1) return i + 1;
		}
		return n + 1;
	}

  private:
};
```

## 总结

$O(1)$的空间限制：

1. 位操作
2. 对称变换操作
3. 本次的交换操作，修改原始数组。

还有中使用hash表存储相应的正数的查询方法，比较简单但是空间复杂度为$O(N)$，不列举了。

## 参考

https://leetcode.com/problems/first-missing-positive/discuss/17071/My-short-c++-solution-O(1
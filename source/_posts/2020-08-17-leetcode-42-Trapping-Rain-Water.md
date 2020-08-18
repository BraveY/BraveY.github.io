---
title: leetcode 42 Trapping Rain Water
date: 2020-08-17 21:20:01
categories: 题解
tags:
- 双指针
- 动态规划
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/trapping-rain-water/)

Given *n* non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

![img](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)
The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. **Thanks Marcos** for contributing this image!

**Example:**

```
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

## 方法1 暴力

### 思路

计算每一格的雨水，然后累加。具体的计算是分别找包含当前节点的左边最大值`max(h[0~i])` 与右边的最大值`max(h[i~n-1])` 之后从这两个最大值中选出最小值，用最小值减去当前的高度，就得到当前的雨水量。
$$
rain[i] = min(max(h[0...i], max(i...n-1))) - h[i]
$$
图片来自花花。

![](https://res.cloudinary.com/bravey/image/upload/v1597718455/blog/coding/lc42bruteForce.jpg)

### 复杂度

时间复杂度$O(N^2)$, 空间复杂度$O(N)$。

### 代码

```cc
/*
TLE O(n^2)
 */
class Solution1 {
  public:
	int trap(vector<int>& height) {
		int n = height.size();
		int ans = 0;
		auto sit = height.begin();
		auto eit = height.end();
		for (int i = 0; i < n; ++i) {
			int l = *max_element(sit, sit + i + 1);//左闭，右开.
			int r = *max_element(sit + i, eit);
			ans += min(l, r) - height[i];
		}
		return ans;
	}

  private:

```

## 方法2 DP

### 思路

主要是对暴力方法对每个格子重复计算左右两边的最大值进行优化。

具体优化方法是用空间换时间，将每个格子左边的最大值用数组`l[i]`存储，右边的最大值用`r[i]`存储。两个数组的更新方法则是简单的一个动态规划。
$$
l[i] = max(h[i], l[i-1])\\
r[i] = max(h[i], r[i+1])
$$

### 复杂度

时间复杂度$O(N)$, 空间复杂度$O(N)$。

### 代码

```cc
/*
Runtime: 12 ms, faster than 66.17% of C++ online submissions for Trapping Rain Water.
Memory Usage: 14.1 MB, less than 57.68% of C++ online submissions for Trapping Rain Water.
 */
class Solution2 {
  public:
	int trap(vector<int>& height) {
		int n = height.size();
		int ans = 0;
		vector<int> l(n);
		vector<int> r(n);
		for (int i = 0; i < n; ++i) {
			l[i] = i == 0 ? height[i] : max(l[i - 1], height[i]);
		}
		for (int i = n - 1; i >= 0; --i) {
			r[i] = i == n - 1 ? height[i] : max(r[i + 1], height[i]);
		}
		for (int i = 0; i < n; ++i) {
			ans += min(l[i], r[i]) - height[i];
		}
		return ans;
	}

  private:
};
```

## 方法2 DP

### 思路

左右两边的数组都是递增数组，因为是取两个最大值的最小值，所以当有一边的最大值已经就是二者之间的最小值的时候就可以更新对应一边的雨水量了。

因为数组是递增的所以可以直接使用两个指针`max_l`和`max_r`来存储左右两边到目前为止的最大值。当左边的最大值小于右边的最大值的时候就计算左边的遍历点雨水量，反之计算右边的雨水量。每次计算完后移动最大值较小的一边的指针，增加值。

图自花花。

![](https://res.cloudinary.com/bravey/image/upload/v1597718456/blog/coding/lc42twoPointers.jpg)

### 复杂度

时间复杂度$O(N)$, 空间复杂度$O(1)$。

### 代码

```cc
/*
Runtime: 12 ms, faster than 66.17% of C++ online submissions for Trapping Rain Water.
Memory Usage: 14.1 MB, less than 65.41% of C++ online submissions for Trapping Rain Water.
 */
class Solution {
  public:
	int trap(vector<int>& height) {
		int n = height.size();
		if (n == 0) return 0;
		int ans = 0;
		int l = 0;
		int r = n - 1 ;
		int max_l = height[l];
		int max_r = height[r];
		while (l < r) {
			if (max_l < max_r) {
				ans += max_l - height[l];
				max_l = max(max_l, height[++l]);
			} else {
				ans += max_r - height[r];
				max_r = max(max_r, height[--r]);
			}
		}
		return ans;
	}

  private:
};
```

## 总结

1. 按照每一格来计算雨水量，而不是纠结于怎么形成U形漏斗
2. 自己考虑的方法如果有很多边界条件，说明方法不对！！
3. 先考虑暴力方法在继续优化

## 参考

[花花leetcode](https://www.bilibili.com/video/BV1hJ41177gG?from=search&seid=10687540897693908584)
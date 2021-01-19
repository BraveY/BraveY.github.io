---
title: leetcode 84 Largest Rectangle in Histogram
date: 2020-09-17 13:24:50
categories: 题解
tags:
- stack
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/largest-rectangle-in-histogram/) 求最大矩形面积。

Given *n* non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.

 

![img](https://assets.leetcode.com/uploads/2018/10/12/histogram.png)
Above is a histogram where width of each bar is 1, given height = `[2,1,5,6,2,3]`.

 

![img](https://assets.leetcode.com/uploads/2018/10/12/histogram_area.png)
The largest rectangle is shown in the shaded area, which has area = `10` unit.

 

**Example:**

```
Input: [2,1,5,6,2,3]
Output: 10
```

## 方法1 暴力

### 思路

遍历所有的两个bar能够形成的矩形面积，从而找到最大的。

### 复杂度

时间复杂度$O(N^2)$

空间复杂度$O(1)$

### 代码

```cc
/*
TLE
 */
class Solution {
  public:
	int largestRectangleArea(vector<int>& heights) {
		int n = heights.size();
		if (!n) return 0;
		int ans = heights[0];
		for (int i = 0; i < n; ++i) {
			int minCur = heights[i];
			if (minCur == 0) continue;
			ans = max(ans, minCur);
			for (int j = i + 1; j < n; ++j) {
				if (heights[j] == 0) {
					break;
				}
				minCur = min(minCur, heights[j]);
				int area = minCur * (j - i + 1);
				ans = max(ans, area);
			}
		}
		return ans;
	}

  private:
};
```

## 方法2 单调栈

### 思路

利用栈来存储递增的区间。如果遍历的高度大于栈顶的坐标高度，则将当前的**索引**（非高度）压入栈，如果当前的高度小于栈顶，则以当前栈顶值作为瓶颈（较低的一段）高度，然后出栈，新栈顶与当前遍历节点之间的距离作为宽度计算所形成的矩形面积。这个面积就是该条形所能形成的最大矩形面积值。

每次开始出栈的时候都是从一个递增区间下降开始的，也就是与右边最大值构成的矩形，只有自己是瓶颈。出栈的时候的高度都是比当前遍历点高度大的，并且高于栈中现有的。

### 复杂度

时间复杂度$O(N)$,空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 24 ms, faster than 86.39% of C++ online submissions for Largest Rectangle in Histogram.
Memory Usage: 14.2 MB, less than 60.71% of C++ online submissions for Largest Rectangle in Histogram.
 */
class Solution2 {
  public:
	int largestRectangleArea(vector<int>& heights) {
		heights.push_back(0);
		const int n = heights.size();
		stack<int> s;
		int ans = 0;
		int i = 0;
		while (i < n) {
			if (s.empty() || heights[i] >= heights[s.top()]) {
				s.push(i++);
			} else {
				int h = heights[s.top()];
				s.pop();
				int w = s.empty() ? i : i - s.top() - 1;
				ans = max(ans, h * w);
			}
		}
		return ans;
	}

  private:
};
```

当出栈之后变为空栈的时候，是遍历到目前为止的最小值，所以直接乘以i就可以了。插入一个0是为了保证一定可以出栈，而不会只入栈。

## 总结

用栈的方法，很巧妙，但是没有总结出规律，看来还是要多刷。

## 参考

[花花](https://zxi.mytechroad.com/blog/stack/leetcode-84-largest-rectangle-in-histogram/)

https://abhinandandubey.github.io/posts/2019/12/15/Largest-Rectangle-In-Histogram.html


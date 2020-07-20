---
title: leetcode 209 Minimum Size Subarray Sum
date: 2020-07-20 14:59:51
categories: 题解
tags:
- 双指针
- 滑动窗口
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/minimum-size-subarray-sum/) 找出和超过指定数字的最小子数组

Given an array of **n** positive integers and a positive integer **s**, find the minimal length of a **contiguous** subarray of which the sum ≥ **s**. If there isn't one, return 0 instead.

**Example:** 

```
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
```

**Follow up:**

If you have figured out the *O*(*n*) solution, try coding another solution of which the time complexity is *O*(*n* log *n*). 

## 方法1 暴力

### 思路

通过以当前遍历的元素为结尾，然后向左遍历做加法，如果和超过了s就返回当前元素结尾的数组长度。在循环中比较出最小的答案。

### 复杂度

两层循环，时间复杂度为$O(n^2)$, 空间复杂度为$O(1)$

### 代码

```cc
/*
Runtime: 628 ms, faster than 5.09% of C++ online submissions for Minimum Size Subarray Sum.
Memory Usage: 10.6 MB, less than 55.91% of C++ online submissions for Minimum Size Subarray Sum.
 */
class Solution2 {
  public:
	int minSubArrayLen(int s, vector<int>& nums) {
		int n = nums.size();
		int ans = INT_MAX;
		for (int i = 0; i < n ; i++) {
			int sum = 0;
			for (int j = i; j >= 0; j--) {
				sum += nums[j];
				if (sum >= s) {
					ans = min(ans, i - j + 1);
					break;
				}
			}

		}
		return ans == INT_MAX ? 0 : ans;
	}

  private:
};
```

## 方法2 滑动窗口

### 思路

使用滑动窗口，当滑动窗口的和小于s的时候右边扩展，一旦窗口的和大于了s，就记录当前长度并和之前的最小答案比较。当 窗口和大于s之后停止向右边扩展，开始从左边逐渐缩小窗口宽度，直至窗口长度刚好小于s。然后又开始向右扩展窗口，直至大于s。

### 复杂度

时间复杂度为$O(N)$,与第一个方法相比，虽然也有两层循环，但是对于第二种方法来说只有少数几个元素需要调整窗口的起始点。第一种方法则是对每个元素都需要重新建立窗口。因此时间复杂度为$O(N^2)$

空间复杂度为$O(1)$

### 代码

```cc
/*
Runtime: 16 ms, faster than 83.14% of C++ online submissions for Minimum Size Subarray Sum.
Memory Usage: 10.5 MB, less than 91.57% of C++ online submissions for Minimum Size Subarray Sum.
 */
class Solution {
  public:
	int minSubArrayLen(int s, vector<int>& nums) {
		int n = nums.size();
		int ans = n + 1;
		int start = 0, end = 0, sum = 0;
		while (start < n) {
			while (sum < s && end < n ) sum += nums[end++]; //右边结尾扩展，增大窗口
			if ( sum  < s) break; // 没有找到大于s的子字符串， 返回0
			ans = min(ans, end - start);
			sum -= nums[start++]; //左边起点向右移动，缩小窗口
		}
		return ans == n + 1 ? 0 : ans;
	}

  private:
};
```

## 参考

https://zxi.mytechroad.com/blog/two-pointers/leetcode-209-minimum-size-subarray-sum/

https://github.com/azl397985856/leetcode/blob/master/problems/209.minimum-size-subarray-sum.md
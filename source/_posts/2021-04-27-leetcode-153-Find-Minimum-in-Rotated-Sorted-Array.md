---
title: leetcode 153 Find Minimum in Rotated Sorted Array
date: 2021-04-27 22:59:17
categories:
tags:
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

旋转数组求出最小值

Suppose an array of length `n` sorted in ascending order is **rotated** between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

- `[4,5,6,7,0,1,2]` if it was rotated `4` times.
- `[0,1,2,4,5,6,7]` if it was rotated `7` times.

Notice that **rotating** an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of **unique** elements, return *the minimum element of this array*.

 

**Example 1:**

```
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
```

**Example 2:**

```
Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
```

**Example 3:**

```
Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
```

 

**Constraints:**

- `n == nums.length`
- `1 <= n <= 5000`
- `-5000 <= nums[i] <= 5000`
- All the integers of `nums` are **unique**.
- `nums` is sorted and rotated between `1` and `n` times.

## 方法1 二分查找

### 思路

可以将数组分为两个递增的阶段，使用指针lo指向第一个阶段最小值，指针hi指向第二个阶段的最大值。

每次根据中间值与hi指针值得比较来判断中间值在第一个递增数组还是第二个递增数组。

移动时如果落在第二个阶段则hi指针移动到mid（不能减1，否则可能到第一阶段去，则回丢失答案），如果落在第一个阶段则lo指针移动到mid+1（因为移动到第二阶段也是有答案的）。

### 复杂度

时间复杂度$O(logN)$
空间复杂度$O(1)$

### 代码

```cc
class Solution {
public:
    int findMin(vector<int>& nums) {
        int len = nums.size();
        if(len==1 || nums[0]<nums[len-1]) return nums[0];
        // return dc_find(nums, 0, len-1);
        return loop_find(nums, 0, len-1);
    }
private:
	int dc_find(vector<int>& nums, int lo, int hi){
		if((hi -lo)==1) return nums[hi];
		int ans = 0;
		int mid = lo + (hi -lo)/2;
		if(nums[mid]>nums[lo]) ans = dc_find(nums, mid, hi);
		else ans = dc_find(nums, lo, mid);
		return ans;
	}
	int loop_find(vector<int>& nums, int lo, int hi){
		int mid = 0;
		while(lo<hi){
			mid = lo + (hi -lo)/2;
			if(nums[mid]>nums[hi]) lo = mid + 1;
			else hi = mid ;
		}
		return nums[lo];
};
```

## 总结

1. 考虑特殊情况旋转过后只有一个递增阶段，则直接返回`nums[0]`
2. 和右边比较作为判断条件
3. 每次移动指针，左边移动时需要移动到mid+1,而右边移动则只移动到mid。
4. 一些具体的小细节判断条件，考虑最简单分治的情况，偶数个或者奇数个，因为循环或者迭代都会经历最小的情况。


## 参考


---
title: leetcode 179 Largest Number
date: 2021-02-03 22:13:12
categories: 题解
tags:
- 排序
- 字符串
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/largest-number/)，使用数组中的元素组合成最大的数字

Given a list of non-negative integers `nums`, arrange them such that they form the largest number.

**Note:** The result may be very large, so you need to return a string instead of an integer.

 

**Example 1:**

```
Input: nums = [10,2]
Output: "210"
```

**Example 2:**

```
Input: nums = [3,30,34,5,9]
Output: "9534330"
```

**Example 3:**

```
Input: nums = [1]
Output: "1"
```

**Example 4:**

```
Input: nums = [10]
Output: "10"
```

 

**Constraints:**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 109`

## 方法1 排序

### 思路

对数组进行特殊的排序，排序的比较函数为两个数字a，b进行字符串组合，ab组合大于ba则a排在b前面，否则相反。

### 复杂度

时间复杂度$O(NLog(N))$

空间复杂度$O(N)$

### 代码

```cc
class Solution {
 public:
	static bool compare(int a, int b) {
		if (a == b) return false;
		string nums1 = to_string(a) + to_string(b);
		string nums2 = to_string(b) + to_string(a);
		int i = 0;
		// while (i < nums1.size()) {
		// 	if (nums1[i] != nums2[i]) {
		// 		return nums1[i] > nums2[i];
		// 		break;
		// 	}
		// 	i++;
		// }
		// return false;
		return nums1 > nums2; // 直接对两个字符串进行比较
	}
	string largestNumber(vector<int>& nums) {
		sort(nums.begin(), nums.end(), compare);//为compare(a,b)为true则a在b前面
		string rtn;
		for (int i = 0; i < nums.size(); i++) {
			rtn += to_string(nums[i]);
		}
		if (rtn[0] == '0') return to_string(0);
		return rtn;
	}

 private:
};
```

## 总结

1. sort()函数的用法，引入的自建比较函数
2. 可以直接对两个构成字符串进行比较
3. to_string()的用法

## 参考
---
title: leetcode 4 Median of Two Sorted Arrays
date: 2020-04-02 20:40:12
categories: 题解
tags:
- 分支
- 二分搜索
copyright: true
---

[题目链接](<https://leetcode.com/problems/median-of-two-sorted-arrays/> )要求求出两个有序数组的总的中位数，但是时间复杂度限制在了$log(m+n)$

There are two sorted arrays **nums1** and **nums2** of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

You may assume **nums1** and **nums2** cannot be both empty.

**Example 1:**

```
nums1 = [1, 3]
nums2 = [2]

The median is 2.0
```

**Example 2:**

```
nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5
```

## 思路

### 双指针

首先最容易想到的方法是采用归并排序中的一个小步骤，使用双指针将两个数组合并后直接可以给出中位数，中间还可以进行计数只用选择到$(m+n)/2$个数字就可以了。这样的方法思路很简单，写起来也很容易，复杂度为$O(m+n)$ 不符合题意，所以跳过。

### 二分搜索

中位数的位置如果是奇数的话是$(m+n+1)/2$记为k,如果是偶数的话则是k和k+1对应的两数之和的均值，这是**第一个条件**。

**第二个条件**则是二分搜索里面的用来比较的条件：左半部分的最大值小于右半部分的最小值。

左半部分中，第一个数组参与了m1个，第二个数组参与了m2个，并且把第一个数组的长度认为小于第二个数组（如果相反，交换下参数的位置即可）。

为了满足第一个条件，即让左半部分的长度为一半的长度，有m1+m2 = k 。这样只用针对第一个数组进行二分就可以了，有了m1的值m2的值自然也确定了。自己之前想的是两个数组都进行二分就没有想出来。

第二个条件需要稍微在转换下，因为左半部分的值最大值是max(nums1[m1-1],nums2[m2-1]),右半部分的最小值是min(nums1[m1],nums2[m2]) 所以分别让
$$
nums1[m1-1]<=nums2[m2],nums2[m2-1]<=nums1[m1]
$$
就可以满足第二个条件了。

这样通过在长度较小的第一个数组中进行二分直到找到符合第二个条件的m1的值就可以输出中位数了。

如果是奇数：max(nums1[m1-1],nums2[m2-1])，如果是偶数(max(nums1[m1-1],nums2[m2-1])+min(nums1[m1],nums2[m2]) )/2。

### 复杂度

O(log(min(m,n))) 

## 代码

判断条件:


$$
nums1[m1-1]<=nums2[m2],nums2[m2-1]<=nums1[m1]
$$

```c++
/*
Runtime: 20 ms, faster than 66.18% of C++ online submissions for Median of Two Sorted Arrays.
Memory Usage: 7.2 MB, less than 100.00% of C++ online submissions for Median of Two Sorted Arrays.
 */
class Solution2 {
  public:
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
		const int n1 = nums1.size();
		const int n2 = nums2.size();
		// make sure n1 < n2, by swapping
		if (n1 > n2) return findMedianSortedArrays(nums2, nums1);
		const int k = (n1 + n2 + 1) / 2 ;
		int l = 0;
		int r = n1;

		while (l <= r) {
			const int m1 = l + (r - l) / 2;
			const int m2 = k - m1;
			const int maxLeftA = m1 <= 0 ? INT_MIN : nums1[m1 - 1];
			const int minRightA = m1 >= n1 ? INT_MAX : nums1[m1];
			const int maxLeftB = m2 <= 0 ? INT_MIN : nums2[m2 - 1];
			const int minRightB = m2 >= n2 ? INT_MAX : nums2[m2];
			if (maxLeftA < minRightB && maxLeftB <= minRightA) {
				// cout << l << "	" << r << endl;
				if ((n1 + n2) % 2 == 1) return max(maxLeftA, maxLeftB);
				else {
					return (max(maxLeftB, maxLeftA) + min(minRightB, minRightA)) * 0.5;
				}
			}
			if (nums1[m1] < nums2[m2 - 1])
				l = m1 + 1;
			else
				r = m1 - 1;
		}
		return 0.0;

	}

  private:
};
```

还有一种代码更简洁的，其方法的条件是寻找maxLeftA>= minRightB的最小的m1值，也就是尽可能的找到二者之间差最小的。

```c++
/*
huahua
Runtime: 20 ms, faster than 66.18% of C++ online submissions for Median of Two Sorted Arrays.
Memory Usage: 7.3 MB, less than 100.00% of C++ online submissions for Median of Two Sorted Arrays.
 */
class Solution1 {
  public:
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
		const int n1 = nums1.size();
		const int n2 = nums2.size();
		// make sure n1 < n2, by swapping
		if (n1 > n2) return findMedianSortedArrays(nums2, nums1);
		const int k = (n1 + n2 + 1) / 2 ;
		int l = 0;
		int r = n1;

		while (l < r) {
			const int m1 = l + (r - l) / 2;
			const int m2 = k - m1;
			// 寻找最小的m1满足 maxLeftA >= minRightB
			if (nums1[m1] < nums2[m2 - 1])
				l = m1 + 1;
			else
				r = m1;
		}
		// cout << l << "	" << r << endl;
		const int m1 = l;
		const int m2 = k - m1;

		const int c1 = max(m1 <= 0 ? INT_MIN : nums1[m1 - 1],
		                   m2 <= 0 ? INT_MIN : nums2[m2 - 1]);

		if ((n1 + n2) % 2 == 1) return c1;

		const int c2 = min(m1 >= n1 ? INT_MAX : nums1[m1],
		                   m2 >= n2 ? INT_MAX : nums2[m2]);
		return (c1 + c2) * 0.5;

	}

  private:
};
```

两种基本一致，有个小细节：

如果二分的时候退出条件是l<r 那么只用移动左边中值加1，右边移动时移动到中值而不是中值减1.

如果循环退出条件时l<=r，那么两边移动左边中值加1，右边中值减1.

## 参考

<https://zhuanlan.zhihu.com/p/39129143> 

<https://zxi.mytechroad.com/blog/algorithms/binary-search/leetcode-4-median-of-two-sorted-arrays/> 

<https://github.com/azl397985856/leetcode/blob/master/problems/4.median-of-two-sorted-array.md> 
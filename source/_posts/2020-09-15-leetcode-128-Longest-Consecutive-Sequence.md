---
title: leetcode 128 Longest Consecutive Sequence
date: 2020-09-15 20:43:12
categories: 题解
tags: 
- 哈希
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/longest-consecutive-sequence/) 寻找最长连续子字符串的长度。

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Your algorithm should run in O(*n*) complexity.

**Example:**

```
Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
```

## 方法1 Online

### 思路

构造kv结构，key是数组值，value是对应key作为边界（包括左边界，与右边界）所构成的连续字串的长度。

动态的将数组值插入到哈希表中，因为是连续的要求，所以去寻找左右两边的邻居节点是否在哈希表中。可以分为三种情况：

1. 没有邻居节点 。 则当前遍历节点的值为1，表示最长只有1.$h[num] = 1$
2. 有一个邻居节点（左/右）。则邻居节点与当前节点的值为二者的和的值+1 $h[num] = h[num\pm1] = h[num]+h[num\pm1]+1$
3. 如果有两个邻居节点，左右都有，则左边邻居与右边邻居的以及当前值更新为当前左邻居与右邻居之和+1 $h[num+1] = h[num -1] = h[num+1]+h[num-1]+1$

总的来说就是分情况然后逐步延长边界，只用维护边界的值就可以了。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 20 ms, faster than 69.33% of C++ online submissions for Longest Consecutive Sequence.
Memory Usage: 11 MB, less than 47.11% of C++ online submissions for Longest Consecutive Sequence.
 */
class Solution1 {
  public:
	int longestConsecutive(vector<int>& nums) {
		unordered_map<int, int> h;
		for (int num : nums) {
			if (h.count(num)) continue;

			auto itL = h.find(num - 1);
			auto itR = h.find(num + 1);
			int l = itL != h.end() ? itL->second : 0;
			int r = itR != h.end() ? itR->second : 0;
			if (l > 0 && r > 0) {
				h[num] = h[num - l] = h[num + r] = l + r + 1; // bridge case
			} else if (l > 0) {
				h[num] = h[num - l] = l + 1;	// one neighbor
			} else if (r > 0) {
				h[num] = h[num + r] = r + 1;	//one neighbor
			} else {
				h[num] = 1; // no neighbor
			}
		}

		int ans = 0;
		for (const auto& kv : h) {
			ans = max(ans, kv.second);
		}
		return ans;
	}

  private:
};
```

## 方法2 offline 

### 思路

先将数组元素存入哈希表中，然后顺序遍历数组，如果元素的左邻居不在哈希表中，则从改元素递增，并记录字符串长度，直到没有元素在哈希表中，说明不再连续。每次遍历数组的时候比较得到最大字符串值。

### 复杂度

时间复杂度$O(N)$，虽然有两个循环，但是每个元素最多访问2次。

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 20 ms, faster than 69.33% of C++ online submissions for Longest Consecutive Sequence.
Memory Usage: 10.8 MB, less than 86.92% of C++ online submissions for Longest Consecutive Sequence.
 */
class Solution {
  public:
	int longestConsecutive(vector<int>& nums) {
		unordered_set<int> h(nums.begin(), nums.end());
		int ans = 0;
		for (long num : nums) {
			if (!h.count(num - 1)) {
				int l = 0;
				while (h.count(num++)) ++l;
				ans = max(ans, l);
			}

		}
		return ans;
	}

  private:
};
```

## 总结

看到线性复杂度的要求，就应该考虑到哈希表。然后剩下的就是怎么设计和维护一个状态了。

## 参考

[花花](https://www.bilibili.com/video/av38647879/)
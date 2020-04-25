---
title: leetcode 55 Jump Game
date: 2020-04-25 12:32:59
categories: 题解
tags:
- 贪心
- 记忆化递归
copyright: true
---

## 题意

[题目链接](<https://leetcode.com/problems/jump-game/> ) 按照数组的值作为步数进行跳跃，能否到达最末端

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

**Example 1:**

```
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**

```
Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum
             jump length is 0, which makes it impossible to reach the last index.
```

### 记忆化递归

### 思路

首先是根据题目的叙述直接暴力的遍历每种可能性，具体的实现方法是用递归。

从后向前考虑每个节点能否到达最后一个节点，对能够达到的节点nums[i]，用递归的思路将nums[0...i]这个数组递归调用后来求这个节点的答案。如果有可行的解直接就返回。

在实现的时候考虑到同一个节点可能会被重复计算很多次，所以使用记忆化递归，用一个hash数组来存储已经计算过的节点，来避免重复计算。

比如[2,3,1,1,4]对第4个节点来说，第一个1节点3和第三个节点1都是可行的，而计算的第三个节点1来说，第1个节点依然是可行的，如果不记录第1个节点3的答案的话，就会重复计算。

### **复杂度**

因为相当于暴力枚举了所有可能的路径，所以复杂度是指数级别的$O(k^n)$ k 是每个节点的可行解。

### 代码

```cc
/*
Runtime: 1204 ms, faster than 5.42% of C++ online submissions for Jump Game.
Memory Usage: 18 MB, less than 5.26% of C++ online submissions for Jump Game.
 */
class Solution2 {
 public:
	bool canJump(vector<int>& nums) {
		int n = nums.size();
		memo[0] = true;
		return helper(nums, n - 1);
	}

 private:
	bool helper(vector<int>& nums, int end) {
		if (memo.count(end)) return memo[end];
		bool ans = false;
		vector<bool> able;
		for (int i = 0; i <= end - 1; i++) {
			if (nums[i] >= end - i) {
				if (ans == true) break;
				ans = ans | helper(nums, i);
			}
		}
		memo[end] = ans;
		return memo[end];
	}
	unordered_map<int, bool> memo;
};
```

## 贪心

更新所有节点能够走的最远值。

### 思路

自己最开始想的是如果每个都选择按照最大的步数走的话，可能出现最大的步数所到的节点值为0，也就无法到达末尾，因此没有想出来具体的贪心方法。

实际上，每个节点选择最大的步数来走在遍历的时候，可以设置一个far变量来记录所有节点可以到达的最远节点。如果far小于了当前节点，说明无法到达当前节点，也就更不能到达末尾。如果遍历完毕后far的大小大于等于n-1，说明可以到达的节点已经超过最后的节点了，可以完成跳跃。

设置了个最远变量后前面说的那种情况也就不存在了，因为只是记录了可能到达的最远值，没有要求只能按照最远的步数来走。

### 复杂度

遍历一次因此$O(n)$ 的复杂度.

### 代码

```cc
/*
Runtime: 8 ms, faster than 93.89% of C++ online submissions for Jump Game.
Memory Usage: 7.8 MB, less than 100.00% of C++ online submissions for Jump Game.
 */
class Solution3 {
 public:
	bool canJump(vector<int>& nums) {
		int n = nums.size();
		int far = nums[0];
		for (int i = 0; i < n; i++) {
			if (i > far) //最远路径无法到达当前位置
				break;
			far = max(far, i + nums[i]);
		}
		return far >= n - 1;
	}

 private:
};
```



## 参考

<https://zxi.mytechroad.com/blog/greedy/leetcode-55-jump-game/> 
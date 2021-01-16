---
title: leetcode 45 Jump Game II
date: 2021-01-15 16:54:36
categories: 题解
tags:
- BFS
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/jump-game-ii/)，到达最末端数组的最短步数。

Given an array of non-negative integers `nums`, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

You can assume that you can always reach the last index.

**Example 1:**

```
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

**Example 2:**

```
Input: nums = [2,3,0,1,4]
Output: 2
```

**Constraints:**

- `1 <= nums.length <= 3 * 104`
- `0 <= nums[i] <= 105`

## 方法1 动态规划

### 思路

dp[i]表示到i节点的最短步数，因为能够到达i节点的路径很多，所以全部遍历之前的节点从而找到能够到达当前节点的并且得到最小值
$$
dp[n] = min(dp[i] + 1)\quad nums[i] >= n - i
$$

### 复杂度

时间复杂度$O(N^2)$ 超时

空间复杂度$O(N)$

### 代码

```cc
/*
O(N^2)
Time Limit Exceeded
*/
class Solution1 {
  public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        if(!n) return 0;
        vector<int> dp(n, INT_MAX);
        dp[0] = 0;
        for(int i = 1; i < n; ++i) {
            for(int j = 0; j < i; ++j) {
                if (nums[j] >= i - j) {
                    dp[i] = min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[n - 1];
    }

  private:
};
```

## 方法2 BFS

### 思路

根据每个点可以确定一个从当前节点出发的最远值区间farthest，也就是其下一层的子节点值。每当遍历到本层的节点最远值end，说明这一层的节点已经遍历完，层数加1，进入下一层，并将下一层的节点end更新为最远可到达节点farthest，知道遍历的节点到达最末端返回当前层数。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 8 ms, faster than 99.94% of C++ online submissions for Jump Game II.
Memory Usage: 15.3 MB, less than 99.91% of C++ online submissions for Jump Game II.
*/
class Solution {
  public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        if(!n) return 0;
        int end = 0, farthest = 0, steps = 0;
        for (int i = 0; i <= end; ++i) {
            if (i == n - 1) break;
            farthest = max(farthest, i + nums[i]);
            if (i == end) {
                ++steps;
                end = farthest;
            }
        }
        return steps;
    }

  private:
};
```

## 总结

1. 通过更新每层的节点可到达的最远区间，来作为下一层的子节点值，与传统BFS的队列记录下一层不一样
2. 存在步数关系可以联想到BFS。
3. DP的方法因为n的规模为$10^4$可以知道$O(N^2)$会超时。

## 参考

[disscuss](https://leetcode.com/problems/jump-game-ii/discuss/?currentPage=1&orderBy=most_votes&query=)
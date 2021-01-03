---
title: leetcode 494 Target Sum
date: 2021-01-03 20:15:42
categories: 题解
tags:
- DFS
- 动态规划
copyright: true
---

# leetcode 494 Target Sum

## 题意

分配操作符号使得数组相应的和为目标数字的分配方式数目。

[题目链接](https://leetcode.com/problems/target-sum/)

You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols `+` and `-`. For each integer, you should choose one from `+` and `-` as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.

**Example 1:**

```
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
```

 

**Constraints:**

- The length of the given array is positive and will not exceed 20.
- The sum of elements in the given array will not exceed 1000.
- Your output answer is guaranteed to be fitted in a 32-bit integer.

## 方法1 DFS

### 思路

将所有的排列全部枚举出来，总共有$2^{20}$种排列组合，排列组合可以联想到DFS。

将每个坐标的值视为一个二叉树节点，其左节点需要加上当前值，右节点需要减去当前值，一直遍历到最末端，如果和为目标和则相应的方法加1.

### 复杂度

时间复杂度$O(2^N)$

空间复杂度$O(N)$。递归调用的栈的空间。

### 代码

```cc
/*
Time Limit Exceeded
*/
class Solution1 {
  public:
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        if (!n) return  0;
        int ans = 0;
        vector<int> prefixSum(n, 0);
        prefixSum[0] = nums[0];
        for(int i = 0; i < n; ++i){
            if(i > 0){
                prefixSum[i] = prefixSum[i - 1] + nums[i];
            }            
        }
        for(int i = 0; i < n; ++i) {
            if(prefixSum[i] == k) ans++;
            for(int j = 0; j < i; ++j){
                if(prefixSum[i] - prefixSum[j] == k) ans++;
            }
        }
        return ans;
    }

  private:
};
```

## 方法2 动态规划

### 思路

数组的和限制为-1000至1000共2001个，取值空间不是很大，所以可以用动态规划。将子问题`dp[i][j]`定义为前i个数组元素和为j的个数，对于`nums[i]`来说，已经知道前一个坐标`i-1`下对应的所有取值下目标和的个数，其从前个坐标转移的状态也就很容易想到，要么是n-1的和加上当前值，要么是减去当前值。
$$
dp[i][j] = dp[i-1][j - nums[i]] + dp[i - 1][j + nums[i]]
$$
因为需要加上一个初始状态`dp[0][j]`表示不用任何数字相加都会有一个目标和为0的情况。所以`dp[i][j]` 的i对应的原始数组为`nums[i-1]`。所以修改状态转移方程为：
$$
dp[i][j] = dp[i-1][j - nums[i - 1]] + dp[i - 1][j + nums[i - 1]]
$$

### 复杂度

时间复杂度$O(N*2001)$

空间复杂度$O(N*2001)$ 因为求`dp[i]`只使用`dp[i-1]`的状态值所以可以只用两列数组，从而优化到$O(2*2001)$

### 代码

```cc
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        int n = nums.size();
		if ( S > 1000 || S < -1000 ) return 0;
		if(!n) return 0;
		vector<vector<int>> dp(2, vector<int>(2001, 0));
		dp[0][0 + 1000] = 1;
		for(int i = 1; i < n + 1; ++i){
			for(int j = 0; j < 2001; ++j){	
				int add = 0;
				int sub = 0;
				if (j - nums[i - 1] >= 0 ) add = dp[i - 1][j - nums[i - 1]];
				if (j + nums[i - 1] < 2001 ) sub = dp[i - 1][j + nums[i - 1]];
				dp[i][j] = add + sub;
			}
		}
		return dp[n][S + 1000];
    }
};
```

空间优化：

```cc
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        int n = nums.size();
		if ( S > 1000 || S < -1000 ) return 0;
		if(!n) return 0;
		vector<vector<int>> dp(2, vector<int>(2001, 0));
		dp[0][0 + 1000] = 1;
		for(int i = 1; i < n + 1; ++i){
			for(int j = 0; j < 2001; ++j){	
				int add = 0;
				int sub = 0;
				if (j - nums[i - 1] >= 0 ) add = dp[0][j - nums[i - 1]];
				if (j + nums[i - 1] < 2001 ) sub = dp[0][j + nums[i - 1]];
				dp[1][j] = add + sub;
			}
			dp[0] = dp[1];
		}
		return dp[1][S + 1000];
    }
};
```

## 总结

1. 动态规划不仅是求最优，还能用来计数。
2. 负数也可以用动态规划，相应的坐标平移就可以了。
3. 二维动态规划是去遍历可能的取值空间。

## 参考

[huahua](https://www.bilibili.com/video/av31501561)
---
title: leetcode 560 Subarray Sum Equals K
date: 2021-01-03 19:10:10
categories: 题解
tags:
- 前缀和
- 哈希表
copyright: true
---

## 题意

连续子数组和为k的个数。

[题目链接](https://leetcode.com/problems/subarray-sum-equals-k/)

Given an array of integers `nums` and an integer `k`, return *the total number of continuous subarrays whose sum equals to `k`*.

**Example 1:**

```
Input: nums = [1,1,1], k = 2
Output: 2
```

**Example 2:**

```
Input: nums = [1,2,3], k = 3
Output: 2
```

**Constraints:**

- `1 <= nums.length <= 2 * 104`
- `-1000 <= nums[i] <= 1000`
- `-107 <= k <= 107`

## 方法1 暴力

### 思路

遍历所有的连续子数组对并直接求和。

### 复杂度

时间复杂度$O(N^3)$

遍历得到所有的子数组对需要$O(N^2)$,对每个字数组求和需要$O(N)$因此三重循环。

空间复杂度$O(1)$

### 代码

略

## 方法2 前缀和

### 思路

A指代数组nums。

使用一个数组`prefixSum[i]`记录到该元素的的和$sum(A[0...i])$,使用`prefixSum[j] - prefixSum[i]`即得到i到j的子数组的和$sum(A[i...j])$。而不用重新再从i开始遍历来求和。

### 复杂度

时间复杂度$O(N^2)$

空间复杂度$O(N)$

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

```

## 方法3 前缀和+哈希表

### 思路

使用一个哈希表来存放已经得到的每个前缀和以及对应的次数，对于`A[i]`其对应的前缀和为`prefixSum[i]`,如果存在前缀和为`prefixSum(j) = prefixSum[i] - k`的元素`A[j]`，则用当前前缀和减去j的前缀和得到的连续子数组`A[j...i]`的和为k
$$
prefixSum(i) - (prefixSum(i) - k) = k
$$
所以只用访问哈希表里面是否存在这样的前缀和，存在则对应的和为K的个数加上该前缀和的出现次数。

相比于方法2遍历的去寻找是否存在，将之前的前缀和存入哈希表后可以直接去寻找对应的前缀和是否存在，从而将查找的时间复杂度从$O(N)$降低到$O(1)$.总的时间复杂度从$O(N^2)$降低到$O(N)$.

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

```cc
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        if(!n) return 0;
        unordered_map<int, int> memo {{0, 1}};
        int prefixSum = 0;
        int ans = 0;
        for(int i = 0; i < n; ++i){
            prefixSum += nums[i];
            ans += memo[prefixSum - k];
            memo[prefixSum]++;
        }
        return ans;
        
    }
};
```

## 总结

1. 求和的题目因为有公式可以转移，所以可以只查找对应的元素是否存在，不用遍历
2. 本题的数组长度和k的取值范围都很大，所以可以排除动态规划。

## 参考

[huahua](https://www.bilibili.com/video/av31350524)
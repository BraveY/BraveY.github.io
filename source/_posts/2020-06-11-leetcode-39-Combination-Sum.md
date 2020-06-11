---
title: leetcode 39 Combination Sum
date: 2020-06-11 15:10:02
categories: 题解
tags:
- 回溯
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/combination-sum/)给出和为指定目标的组合，不能重复。

Given a **set** of candidate numbers (`candidates`) **(without duplicates)** and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

The **same** repeated number may be chosen from `candidates` unlimited number of times.

**Note:**

- All numbers (including `target`) will be positive integers.
- The solution set must not contain duplicate combinations.

**Example 1:**

```
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
```

**Example 2:**

```
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

## 回溯法

### 思路

列举出所有的组合，所以很自然地想到回溯法。这里变化的地方则是状态变量是path之和以及开始的节点索引值index。因为每个节点的下一步是所有的可选值，因此不需要用[46排列组合](https://bravey.github.io/2020-05-29-leetcode-46-Permutations.html) 中的used状态变量。但因为是不能有重复的组合出现，所以下一步的取值只考虑已被选中节点的后半部分，也就是从上一个进入path的节点值开始。可以理解成后面的进入path的节点，如果是之前已经走过的节点，那么在之前的循环遍历中已经把这个节点符合条件的所有路径都穷举出来了，所以不用在考虑之前已经走过的节点路径了。但本题可以重复地选择一个节点作为组合，所以开始的索引index就是当前最新加入path的节点索引值.

另外一个点就是进行剪枝的一个操作，比如如果加入当前节点的sum值已经超过target了，那么就跳过当前节点的所有子节点，因为后续的加入的节点肯定会超过target。

总结下，状态变量为两个：path的sum，已经当前新加入path的节点的索引值。以及剪枝操作。

### 复杂度

如果不剪枝的话，应该是个无限递归，因为可以一直选择同一个节点进行递归。但是剪枝了之后的时间复杂度，有点不好分析。

### 代码

```cc
/*
Runtime: 4 ms, faster than 98.97% of C++ online submissions for Combination Sum.
Memory Usage: 11.1 MB, less than 53.21% of C++ online submissions for Combination Sum.
 */
class Solution {
  public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        backtrack(candidates, target, 0, 0);
        return ans;
    }

  private:
    vector<vector<int>> ans;
    vector<int> path;
    void backtrack(vector<int>& candidates, int target, int sum, int index) {
        for (int i = index; i < candidates.size(); i++) {
            if (sum + candidates[i] > target) continue;
            path.push_back(candidates[i]);
            sum += candidates[i];
            if (sum == target) {
                ans.push_back(path);
            } else {
                backtrack(candidates, target, sum, i);
            }
            path.pop_back();
            sum -= candidates[i];
        }
    }
};
```

## 参考

https://www.bilibili.com/video/av48573740/
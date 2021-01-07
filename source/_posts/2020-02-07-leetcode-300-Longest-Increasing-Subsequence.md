---
title: leetcode 300 Longest Increasing Subsequence
date: 2020-02-07 21:19:32
categories: 题解
tags:
- 动态规划
- 二分查找
copyright: true
---

# leetcode 300 Longest Increasing Subsequence

[题目来源](<https://leetcode.com/problems/longest-increasing-subsequence/> )，需要求解数组中的升序子数组的最大长度。

<!--more-->

## 思路

### 动态规划

### 自己的思路

输入是数组，求解的也是最优化问题，所以首先考虑动态规划。我自己的思路是使用背包问题的解法，从长度为n的数组A[0...n-1]中，选择数字构成增序子序列。背包的容量限制就是升序数列的最后一个元素，所以初始的限制是最大值，这样任意一个数字都是在升序的。

所以定义最优子结构$opt[i][j]$表示：前i个元素中，最大值不超过第nums[j]的最大长度。如果选了第i个元素那么状态为$opt[i-1][i]+1$,不选的话状态从$opt[i-1][j]$转移来。这样可以写出状态转移表达式：$opt[i][j]=max(opt[i-1][i]+1,opt[i-1][j])$

#### 复杂度

c需要两层循环，所以为$O(n^2)$使用了二维数组所以空间复杂

### 更优的方法

将最优子结构定义为**必须以第i个数字结尾**nums[i]的最长增长子数组的长度，加上这个限制后就可以简单的根据选或者不选的原则来进行状态转移了。对第i个数字nums[i]，其最优结果是前面i个数组中**每个数字**nums[j]的结果加一的最大值。也就是对前面每个元素都遍历一遍来确定其能够达到的最大值。状态转移表达式为：$opt[i] =max({opt[i],opt[j]+1}) j<=i$,这时候的子结构的解并不是原始问题的解，要对opt数组求解最大值才能够得到原始问题的解。整个子问题的定义类似于[leetcode 53](<https://bravey.github.io/2020-01-29-leetcode-53-Maximum-Subarray.html> )的定义方法。

状态转移是与前面所有的数字进行遍历转移而来的，与普通常见的只从某几个状态转移而来不同，可以从前面所有的状态转移而来，（更进一步剪枝的话只从比当前要小的状态转移而来，不过因为会记录最大值所以可以不用剪枝）

#### 复杂度

需要两层循环，所以时间复杂度为$O(n^2)$使用了一维数组所以空间复杂度$O(n)$

### 记忆化递归

递归的方法,可以做分治的思路去考虑。最开始的问题是要求解LIS[10,9,2,5,3,7,101,18],那么从最后一位元素18开始考虑，选择了18作为最后一个元素的时候，需要从末尾元素小于18的子数组中寻找答案。也就是
$$
LIS[10,9,2,5,3,7,101,18] = max(LIS[10,9,2,5,3,7]+1,LIS(10,9,2,5,3)+1,LIS[10,9,2,5]+1,LIS[10,9,2]+1,LIS[10,9]+1,LIS[10]+1)
$$
为了避免同样的子问题反复被计算，因此使用记忆化递归的方法，将每个子问题的答案记录在对应的数组元素中，递归的退出条件有个查表。

依然需要两个循环，从数组由小到大进行计算。

### 复杂度

需要两层循环，所以时间复杂度为$O(n^2)$使用了一维数组所以空间复杂度$O(n)$

### 二分查找

思路是建立一个一维数组opt，按照顺序读入原来的数组元素nums[i]，如果nums[i]比opt的首元素小，则更新首元素为nums[i]，如果比opt末尾元素还要大则在opt数组后面插入nums[i],如果nums[i]大于首元素却小于末尾元素，使用二分查找第一个大于等于nums[i]的数字，并将其更新为nums[i]。

opt数组是一个递增数组，nums[i]每次都会更新进opt数组中。

使用二分查找寻找在opt数组中第一个大于等于它数字。如果大于等于nums[i]的第一个数字存在就更新这个数字为nums[i]，如果不存在就在opt末尾增加这个元素。遍历完后返回opt数组的长度作为答案，需要注意opt数组存储的值并不一定对应着真实的LIS。

之所以这样做能够得到答案，是因为记录了遍历以来出现过的最大值的个数，相当于是把增长的波峰给记录了下来，相当于是打破记录的次数，而打破记录的值一定是增序的。

### 复杂度

二分查找，所以时间复杂度为$O(nlogn)$使用了一维数组所以空间复杂度$O(n)$

## 代码

完整代码文件在[github](<https://github.com/BraveY/Coding/blob/master/leetcode/300longest-increasing-subsequence.cpp> )上面。

## 动态规划

### 自己的

```cc
/*
Runtime: 184 ms, faster than 5.32% of C++ online submissions for Longest
Increasing Subsequence. Memory Usage: 110.9 MB, less than 6.25% of C++ online
submissions for Longest Increasing O(n^2) 时间和空间
 */
class Solution {
 public:
  int lengthOfLIS(vector<int>& nums) {
    nums.push_back(INT_MAX);
    int n = nums.size();
    vector<vector<int> > opt(n + 1, vector<int>(n + 1, 0));
    for (int i = 1; i <= n - 1; i++) {
      for (int j = 1; j <= n; j++) {
        if (nums[i - 1] < nums[j - 1])
          opt[i][j] = max(opt[i - 1][i] + 1, opt[i - 1][j]);
        else
          opt[i][j] = opt[i - 1][j];
      }
    }
    return opt[n - 1][n];
  }

 private:
};
```

## 更优的

```cc
class Solution3 {
 public:
  int lengthOfLIS(vector<int>& nums) {
    if (nums.empty()) return 0;
    int n = nums.size();
    auto opt = vector<int>(n, 1);
    for (int i = 1; i < n; ++i)
      for (int j = 0; j < i; ++j)
        if (nums[i] > nums[j]) opt[i] = max(opt[i], opt[j] + 1);
    return *max_element(opt.begin(), opt.end());
  }
};
```

## 记忆化递归

```cc
class Solution2 {
 public:
  int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    memo = vector<int>(n, 0);
    int ans = 0;
    for (int i = 0; i < n; ++i) ans = max(ans, LIS(nums, i));
    return ans;
  }

 private:
  vector<int> memo;
  // length of LIS ends with nums[r]
  int LIS(const vector<int>& nums, int r) {
    if (r == 0) return 1;
    if (memo[r] > 0)
      return memo[r];  // 记忆化递归
                       // 退出的条件还有个查表，如果已经计算过则直接返回
    int ans = 1;
    for (int i = 0; i < r; ++i)
      if (nums[r] > nums[i]) ans = max(ans, LIS(nums, i) + 1);
    memo[r] = ans;
    return memo[r];
  }
};
```

## 二分查找

```cc
/*
https://www.cnblogs.com/grandyang/p/4938187.html
Runtime: 0 ms, faster than 100.00% of C++ online submissions for Longest
Increasing Subsequence. Memory Usage: 8.7 MB, less than 62.50% of C++ online
submissions for Longest Increasing Subsequence.
 */
class Solution4 {
 public:
  int lengthOfLIS(vector<int>& nums) {
    if (nums.empty()) return 0;
    vector<int> opt;
    for (int i = 0; i < n; i++) {
      int lo = 0, hi = opt.size();
      while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (opt[mid] < nums[i])
          lo = mid + 1;
        else
          hi = mid;
      }
      if (hi >= opt.size())
        opt.push_back(nums[i]);
      else
        opt[hi] = nums[i];
    }
    return opt.size();
  }
};
```

## 参考

[记忆化递归与更优的DP解法](<https://www.youtube.com/watch?v=7DKFpWnaxLI&feature=youtu.be> )

[二分查找的解法](https://www.cnblogs.com/grandyang/p/4938187.html)
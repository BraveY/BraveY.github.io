---
title: leetcode 53 Maximum Subarray
date: 2020-01-29 14:33:50
categories: 题解
tags:
- leetcode
- 动态规划
- 分治
copyright: true
---

# leetcode 53 Maximum Subarray

[题目来源](<https://leetcode.com/problems/maximum-subarray/> )。题意要求给出一个数组中拥有最大和的连续子数组。

<!--more-->

## 思路

### 枚举

最简单的思路，连续的数组，只要枚举出所有的头和尾的索引对，然后计算这些所有枚举对中的和，然后输出这些和之中的最大和就可以了。

### 复杂度

因为总共有$1+2+3...+n=\frac{n(n+1)}{2}$ 对，然后每个对需要花费O(n)的时间去求和，因此总共的时间复杂度为$O(n^3)$.

### 分治

观察的输入的数据是数组，数组是典型的可分的数据结构，因此可以考虑使用分治法。首先定义子问题为：长度为n的数组$A[0...n-1]$的连续子数组的最大和。从中间元素$A[\frac{n}{2}]$划分成左右两个数组，变成左边$L[0...\frac{n}{2}]$与右边$R[\frac{n}{2}+1...n-1]$两个子问题。划分成两个自问题后，考虑最大和的连续数组要么在左边数组，要么在右边数组，还有一种情况是连续数组横跨中间的元素，即占了左边的一部分，又占了右边的一部分。这种交叉的情况，就是左边数组必须以中间元素$A[\frac{n}{2}]$为结尾的最大连续数组和，与右边必须以$A[\frac{n}{2}+1]$为开始的最大连续数组和。这种交叉的情况只能通过左右两边分别遍历来得到结果。这种情况的分析与493[逆序对的计数](<https://bravey.github.io/2019-10-07-%E9%80%86%E5%BA%8F%E5%AF%B9%E7%9A%84%E8%AE%A1%E6%95%B0.html> )的分支情况很相似。

**Divide** 将输入数组A划分为左边A[0, n/2] 与右边A[n/2+1, n-1]两个数组 

**Conquer** 对左边与右边的子数组递归调用求解

**Merge **遍历求交叉的情况，然后与左边和右边的最大连续和三者比较求得最大值并返回

一个形象的[参考](<https://github.com/azl397985856/leetcode/blob/master/problems/53.maximum-sum-subarray-cn.md> )图示：

![](https://github.com/azl397985856/leetcode/raw/master/assets/problems/53.maximum-sum-subarray-divideconquer.png)



### 复杂度

与暴力求解需要遍历$n^2$个索引对相比，分治法只需要遍历递归树的深度$log(n)$次。

将规模为n的问题分解成两个两个n2n2 的子问题问题，同时对两个子问题进行合并的复杂度为O(n)，所以有递推公式： 
$$
T(n)=\left\{
  \begin{array}{**lr**}  
             1 &  n=1\\\\  
             2T(\frac{n}{2})+O(n)
             \end{array}  
\right.
$$
根据主定理有最后的复杂度为O(nlog(n))  

### 动态规划

观察题意，求解的是最大这种求最优的问题，然后问题又可分，因此可以考虑动态规划的方法。动态规划最重要的就是最优子结构的定义。我当时做的思路就是把最优子结构定义为分治的子问题：长度为n的数组的最大和用OPT[n-1]表示。然后考虑最后一个末尾元素是否在连续数组中来考虑，考虑状态OPT[n-1]是怎么从OPT[n-2]来转移的。从这个思路就会陷入一种困难的情况，因为去除最后一个元素A[n-1]后，OPT[n-1]与OPT[n-2]状态转移不是max(OPT[n-1],OPT[n-2]+A[n-1])。因为要求连续，所以可能要加上前面其他的元素。比如以[1,-3, 4]来举例，4如果在连续数组中，并不是OPT[1,-3]+4，而是OPT[1,-3]+(-3)+4。

从前面这种思路可以得到一个启发，如果要使用OPT[n-2]+A[n-1]来表示状态的转移，需要限定OPT[n-1]为第n-1个元素必须参与到连续数组中。所以定义最优子结构OPT[n]为：**第n个元素一定参与的连续数组最大和。**这个时候就可以使用max(OPT[n-1],OPT[n-2]+A[n-1])来进行状态转移，这个max(OPT[n-1],OPT[n-2]+A[n-1])的状态转移方程也可以用一个if语句来判断，如果OPT[n-2]为负数就不用加上A[n-1],因为加上一个负数只会使得连续和变小。

根据这个子结构的定义，对于样例[-2 1 -3 4 -1 2 1 -5 4]，其对应的状态数组OPT为：[-2 1 -2 4 3 5 6 1 5]。即对于第2个元素-3，以其结尾的连续数组中，最大和为-2。这个子结构并不是对应问题的原始解，原始解是对应的OPT数组截止目前的最大值。即：maxSum[n-1]=max(OPT[n-1],maxSum[n-1])。

[参考](<https://github.com/azl397985856/leetcode/blob/master/problems/53.maximum-sum-subarray-cn.md> )图示:

![](https://github.com/azl397985856/leetcode/raw/master/assets/problems/53.maximum-sum-subarray-dp.png)

### 复杂度

因为只需要遍历一遍就可以把OPT数组给填充好，所有时间复杂度为O(n)，因为每次状态转移只依赖于前面一个状态，所以可以只使用O(1)的空间。

## 代码

源码文件在[github](<https://github.com/BraveY/Coding/blob/master/leetcode/53maximum-subarray.cpp> )上。

### 分治

使用坐标索引来传递。

```cc
/*
DC解法
Runtime: 948 ms, faster than 5.13% of C++ online submissions for Maximum
Subarray. Memory Usage: 9.6 MB, less than 5.88% of C++ online submissions for
Maximum Subarray. Maximum Subarray. 时间复杂度O(nlogn),空间复杂度O(1)/O(logn)
 */
class Solution3 {
 public:
  int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    if (0 == n) return INT_MIN;
    if (1 == n) return nums[0];
    return helper(nums, 0, n - 1);
  }

 private:
  int helper(vector<int>& nums, int lo, int hi) {
    if (lo >= hi) return nums[lo];
    int mid = lo + (hi - lo) / 2;
    int left = helper(nums, lo, mid);
    int left_max = nums[mid];
    int left_sum = 0;
    for (int i = mid; i >= 0; i--) {
      left_sum += nums[i];
      left_max = max(left_sum, left_max);
    }
    int right = helper(nums, mid + 1, hi);
    int right_max = nums[mid + 1];
    int right_sum = 0;
    for (int i = mid + 1; i <= hi; i++) {
      right_sum += nums[i];
      right_max = max(right_sum, right_max);
    }
    int cross = right_max + left_max;
    int max_all = max(max(left, right), cross);
    return max_all;
  }
};
```

### 动态规划

第一种使用一维数组

```cc
/*
DP解法
Runtime: 8 ms, faster than 72.96% of C++ online submissions for Maximum
Subarray. Memory Usage: 9.4 MB, less than 18.63% of C++ online submissions for
Maximum Subarray.
时间复杂度O(n),空间复杂度O(n)
 */
class Solution1 {
 public:
  int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    int ans = nums[0];
    std::vector<int> opt(n, INT_MIN);
    for (int i = 1; i < n; i++) {
      opt[i] = opt[i - 1] > 0 ? nums[i] + opt[i - 1] : nums[i];
      ans = max(opt[i], ans);
    }
    return ans;
  }

 private:
};
```

第二种不使用一维数组

```cc
/*
DP解法
Runtime: 4 ms, faster than 98.48% of C++ online submissions for Maximum
Subarray. Memory Usage: 9.3 MB, less than 74.51% of C++ online submissions for
Maximum Subarray. 时间复杂度O(n),空间复杂度O(1)
 */
class Solution2 {
 public:
  int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    int ans = nums[0];
    int opt = nums[0];
    for (int i = 1; i < n; i++) {
      opt = opt > 0 ? nums[i] + opt : nums[i];
      ans = max(opt, ans);
    }
    return ans;
  }

 private:
};
```



## 参考

azl397985856的题解以及文中图示的参考[链接](https://github.com/azl397985856/leetcode/blob/master/problems/53.maximum-sum-subarray-cn.md)

[花花leetcode](<https://www.youtube.com/watch?v=7J5rs56JBs8&feature=youtu.be> ) 53题题解 
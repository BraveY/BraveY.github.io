---
title: 阿里笔试combinationSum
date: 2021-03-18 15:19:23
categories:
tags:
copyright: true
---

## 题意

n副卡牌，每一副卡牌有m张牌，编号为1到m，问从每幅卡牌中抽出一张，使得选出的卡牌编号相加之和为k，总共有多少种方案。答案很大，需要对$10^9+7$取余。

## 方法1 回溯

### 思路

相当于枚举所有的排列组合。

将n个数组理解为n层节点，每层总共有m个节点，当遍历到叶子节点的时候判断和是否为k。

中途可以进行剪枝。即当前和大于目标和则返回，以及剩下的每层选则最大的数字m，也不能凑够剩余的和的时候直接返回。

类似于[leetcode 39 combinationSum](https://leetcode.com/problems/combination-sum/)

### 复杂度

时间复杂度$O(N^M)$

空间复杂度$O(1)$

### 代码

```cc
/*
n个数组作为n层节点。
backtrack 来求解所有的路径。
*/
class Solution2 {
public:
    int solve(int n, int m, int k) {
        ans = 0;
        dfs(0, 0, m, n, k);
        return ans;
    }

    int ans = 0;
    void dfs(int sum, int cur, int m, int n, int k) { //cur 作为depth
        if (sum > k) return; // 有个简单的剪枝，当前sum已经超过了目标就直接返回
        if (k - sum > (n - cur)*m) return ; // 剩下的全选最大的m也凑不够需要的，直接返回
        if (cur == n) {
            if (sum == k) // 
                // cout << "m: " << m << "n: " << n << endl;
                ans = (ans % MOD) + 1;
            return;
        }
        for (int i = 1; i <= m; i++) {
            sum += i;
            dfs(sum, cur + 1, m, n, k);
            sum -= i;
        }
    }
};
```

## 方法2 DP

### 思路

因为前面的选择影响到后面的选择，存在状态转移，所以可以联想用动态规划。类似于交换硬币的问题。将子问题`dp[n][k]` 定义为从n副牌里面选择出和为k的方案个数，状态转移方程考虑最后一幅牌能够出的元素，列举所有情况然后累加和。
$$
dp[n][k] += dp[n-1][k-i] \quad i \in [1...m]
$$


### 复杂度

时间复杂度$O(NMK)$

空间复杂度$O(NK)$

### 代码

```cc
class Solution {
  public:
	int combinationSum(int n, int m, int k) {
        vector<vector<int>> dp(n+1, vector<int>(k+1, 0));
        for (int i = 1; i <= k; ++i) {
            if (i <= m) dp[1][i] = 1;
        }
        for (int i = 2; i <= n; ++i){
            for (int j = 1; j <= k; ++j) {
                for (int p = 1; p <= j; ++p) {     // 和为j的个数 当前数组出p的情况 考虑p大于m的情况
                    if(p <= m ) {  // j可能大于m，但是p的范围是1至m    
                        dp[i][j] +=dp[i-1][j-p];
                        dp[i][j]  %= MOD;  // 确保每次组合得到的值都小于MOD
                    }
                }                
            }
        }
        return dp[n][k] % MOD;
    }

  private:
};
```

## 总结

1. 动态规划的代码考虑当前数组能够出的元素范围。

## 参考


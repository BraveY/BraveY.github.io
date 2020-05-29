---
title: leetcode 46 Permutations
date: 2020-05-29 11:41:39
categories: 题解
tags:
- 回溯
- DFS
copyright: true
---

## 题意

[题目链接](<https://leetcode.com/problems/permutations/> ) 求给定数字的所有排列组合

Given a collection of **distinct** integers, return all possible permutations.

**Example:**

```
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

## 方法

使用回溯法，但是之前没有做过这个类型的题，所以记录下。

### 思路

从树的角度来考虑，第0层是[]空节点，没有使用过的数字，所以所有的数字都是他的子节点因此第一层有3个子节点即`[1],[2],[3]`，然后进入第二层的第一个节点，已经使用过了1这个数字，剩下的只有2和3两个节点所以第二层的第一个节点的子节点有两个，`[1,2],[1,3]` 在第三层左边第一个子节点因为已经使用过了[1,2]两个节点所以，只有`[1,2,3]`一个节点。

![](https://res.cloudinary.com/bravey/image/upload/v1590726036/blog/coding/lc46.jpg )

树的构成很容易想到，关键是怎么具体的遍历，因为树的构成不是二叉树那样直接通过指针指向下一个节点的。这里的寻找下一个节点的思路就是暴力for循环，如果这个数字没有被使用那么就是当前节点的子节点，然后对当前子节点继续递归调用深度遍历。因此需要使用一个`used`数组来记录当前节点已经使用过的数字，使用一个`path`数组来存储已经走过的路径。单独用一个used数组可以直接`O(1)`地去查询是否使用过。当走到最底层，也就是`depth==len`的时候就可以把这个节点的`path`加入到最终要返回的`ans`二维数组中。

所谓回溯，就是当从子节点遍历返回以后，会将**状态变量重置**为刚进入当前节点的状态。这样就可以考虑下一个节点的遍历了。

### 复杂度

时间复杂度考虑所有的节点数 $O(n*n!)$ 详细推导见参考1。

空间复杂度为$O(n*n!)$， 因为在叶子节点拷贝$O(n)$ path，而总共有$O(n!)$ 个叶子节点。但不是一次性需要的，因为是递归的最后一层才进行的拷贝。

### 代码

```cc
/*
Runtime: 4 ms, faster than 92.44% of C++ online submissions for Permutations.
Memory Usage: 7.9 MB, less than 100.00% of C++ online submissions for Permutations.
 */
class Solution {
  public:
	vector<vector<int>> permute(vector<int>& nums) {
		int len = nums.size();
		vector<vector<int>> ans;
		if (len == 0) return ans;
        //状态变量
		vector<int> path;
		vector<bool> used(len);
		backtrack(nums, len, 0, path, used, ans);
		return ans;
	}

  private:
	void backtrack(vector<int>& nums, int len, int depth,
	               vector<int>& path, vector<bool>& used, vector<vector<int>>& ans) {
		if (depth == len) { //到达叶子节点
			ans.push_back(path);
			return;
		}
		for (int i = 0; i < len; i++) {
			if (used[i]) continue;
			used[i] = true;//先标记为使用，供子节点使用
			path.push_back(nums[i]);
			backtrack(nums, len, depth + 1, path, used, ans);
            //子节点递归调用之后，回溯，将状态重置，进入下一个子节点的访问
			path.pop_back();
			used[i] = false;
		}
	}
};
```

## 参考

[参考1 官方题解](<https://www.bilibili.com/video/BV1oa4y1v7Kz?from=search&seid=13759472018639491202> )

[参考2 花花](<https://zxi.mytechroad.com/blog/searching/leetcode-46-permutations/> )
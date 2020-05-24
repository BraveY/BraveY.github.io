---
title: leetcode 98 Validate Binary Search Tree
date: 2020-05-24 15:16:33
categories: 题解
tags:
- DFS
copyright: true
---

## 题意

[题目链接](<https://leetcode.com/problems/validate-binary-search-tree/> ) 要求判断一个二叉树是否是二叉搜索树

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

- The left subtree of a node contains only nodes with keys **less than** the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

 

**Example 1:**

```
    2
   / \
  1   3

Input: [2,1,3]
Output: true
```

**Example 2:**

```
    5
   / \
  1   4
     / \
    3   6

Input: [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
```

## 方法

自己之前的做法只考虑到了子节点的值与父节点比较，这样只考虑了左边的最大，没有考虑到左边的最小边界。同样的右边也只考虑了最小边界而没考虑到最大边界。

### 方法1：边界递归传递

对每个节点考虑其最大与最小边界，超过了这个边界就不行了。边界的传递规律是，左边的节点将最大边界`max_limit`更新为父节点的值，而最小边界`min_limit` 与父节点保持一致。右子节点则是更新最小边界`min_limit`为父节点的值，最大边界`max_limit`与父节点保持一致。然后用DFS的方法递归调用。

### 复杂度

时间和空间复杂度都是$O(n)$

### 代码

```cc
/*
Runtime: 16 ms, faster than 70.05% of C++ online submissions for Validate Binary Search Tree.
Memory Usage: 21.6 MB, less than 5.21% of C++ online submissions for Validate Binary Search Tree.
 */
class Solution1 {
  public:
	bool isValidBST(TreeNode* root) {
		return isValidBST(root, LLONG_MIN, LLONG_MAX);
	}

  private:
	bool isValidBST(TreeNode* root, long min_limit, long max_limit) {
		if (!root) return true;
		if (root->val <= min_limit || root->val >= max_limit)
			return false;
		return isValidBST(root->left, min_limit, root->val) &&
		       isValidBST(root->right, root->val, max_limit);
	}
};
```

### 方法2：中序遍历判断是否有序

这个的思路就是二叉搜索数的中序遍历的结果是升序排序了的，所以在遍历的时候判断当前节点的值是否比前一个值小，是的话说明并非二叉搜索树。

### 复杂度

时间和空间复杂度都是$O(n)$

### 代码

```cc
/*
Runtime: 24 ms, faster than 19.28% of C++ online submissions for Validate Binary Search Tree.
Memory Usage: 21.6 MB, less than 5.21% of C++ online submissions for Validate Binary Search Tree.
 */
class Solution {
  public:
	bool isValidBST(TreeNode* root) {
		prev = nullptr;
		return inOrder(root);
	}

  private:
	TreeNode* prev;
	bool inOrder(TreeNode* root) {
		if (!root) return true;
		if (!inOrder(root->left)) return false;
		if (prev && root->val <= prev->val) return false;
		prev = root;
		return inOrder(root->right);
	}
};
```

## 参考

[花花](<https://www.bilibili.com/video/av38708037/> )


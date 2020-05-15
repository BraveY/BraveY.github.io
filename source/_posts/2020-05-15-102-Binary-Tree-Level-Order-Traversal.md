---
title: 102 Binary Tree Level Order Traversal
date: 2020-05-15 11:41:58
categories: 题解
tags:
- BFS
- DFS
copyright: true
---

## 题意

[题目链接](<https://leetcode.com/problems/binary-tree-level-order-traversal/> ),把二叉树的每个深度节点从左到右给记录在二维数组中。

Given a binary tree, return the *level order* traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

return its level order traversal as:

```
[
  [3],
  [9,20],
  [15,7]
]
```

## BFS

BFS一般都是用队列来实现的，但这里不能用队列，因为用队列无法记录每层的信息。

为了记录每层的信息需要使用**两个数组**，一个数组cur记录当前层需要遍历的节点信息，另一个数组next记录下一层需要遍历的节点信息。两个数组不断交换来完成队列的出队入队工作。因为是划分了层次的，所以可以很方便的记录每层的节点。

### 代码

```c++
/*
BFS used two vectors
Runtime: 0 ms, faster than 100.00% of C++ online submissions for Binary Tree Level Order Traversal.
Memory Usage: 12.4 MB, less than 100.00% of C++ online submissions for Binary Tree Level Order Traversal
 */
class Solution1 {
 public:
	vector<vector<int>> levelOrder(TreeNode* root) {
		if (!root) return {};
		vector<vector<int>> ans;
		vector<TreeNode*> cur, next;
		cur.push_back(root);
		while (!cur.empty()) {
			ans.push_back({}); //直接用{}，就可以了，自动初试为空vector
			for (auto node : cur) {
				ans.back().push_back(node->val);//back()返回vector的最后一个元素，二维数组末尾的插入
				if (node->left) next.push_back(node->left);
				if (node->right) next.push_back(node->right);
			}
			swap(cur, next);
			next.clear();
		}
		return ans;
	}
};
```

### 复杂度

O(N) N为节点数。

## DFS

因为需要记录深度信息，所以每次递归调用的时候传递一个深度就可以了。

实现上需要注意，二维数组可能没有depth那么长，所以用while循环来插入空数组，以保证长度。

### 代码

```cc
/*
DFS
Runtime: 4 ms, faster than 94.10% of C++ online submissions for Binary Tree Level Order Traversal.
Memory Usage: 13.8 MB, less than 90.14% of C++ online submissions for Binary Tree Level Order Traversal.
 */
class Solution {
 public:
	vector<vector<int>> levelOrder(TreeNode* root) {
		if (!root) return {};
		ans.push_back({});
		DFS(root, 0);
		return ans;
	}
 private:
	vector<vector<int>> ans;
	void DFS(TreeNode* node, int depth) {
		if (!node) return;
		while (ans.size() <= depth ) {
			ans.back().push_back({});
		}
		ans[depth].push_back(node->val);
		DFS(node->left, depth + 1);
		DFS(node->right, depth + 1);
	}
};
```

### 复杂度

O(N) N为节点数。

## 参考

<https://zxi.mytechroad.com/blog/leetcode/leetcode-102-binary-tree-level-order-traversal/> 
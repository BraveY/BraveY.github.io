---
title: 102 Binary Tree Level Order Traversal & 103 
date: 2020-05-15 11:41:58
categories: 题解
tags:
- BFS
- DFS
copyright: true
---

# 102

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

卡壳的地方是怎么在BFS中记录层数的信息。

## BFS

~~BFS一般都是用队列来实现的，但这里不能用队列，因为用队列无法记录层数的信息~~。注意这里可以用队列来记录，只是单纯的一个队列是无法记录节点的层数信息,也需要两个队列来记录，实现方式见103题解。

为了记录每层的信息需要使用**两个数组**，一个数组cur记录当前层需要遍历的节点信息，另一个数组next记录下一层需要遍历的节点信息。两个数组不断交换来完成队列的出队入队工作。因为是划分了层次的，所以可以很方便的记录每层的节点。

### 代码1 双数组

代码逻辑：使用cur和next两个数组，记录上一层和下一层的节点地址。

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
			for (auto node : cur) {//遍历的时候就可以将每一层的节点push进ans中。
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

## 总结

1. 使用BFS遍历时为了记录层数的信息，需要使用**2个数组或者2个队列**来记录信息，一个时当前层次的，另一个时下一个层次的。
2. 使用DFS则需要提前把返回的二维数组建立起来方便直接根据深度来插入对应的点。

## 参考

<https://zxi.mytechroad.com/blog/leetcode/leetcode-102-binary-tree-level-order-traversal/> 

# 103

## 题意

[题目链接](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/ ),把二叉树之字形遍历。

Given a binary tree, return the *zigzag level order* traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```



return its zigzag level order traversal as:

```
[
  [3],
  [20,9],
  [15,7]
]
```

## 思路

思路同上，面试的时候想到用两个队列来实现，但是没有写出来，后面下来重写的时候发现，双队列写的时候会比较绕,推荐双数组。

## 代码1 双队列

1. 先将左右子节点入到下层队列。
2. tmp数组反向的reverse函数用法：`reverse(tmp.begin(), tmp.end())`。
3. 交换q1与q2可以直接用swap函数。

```cc
/*
Runtime: 4 ms, faster than 73.92% of C++ online submissions for Binary Tree Zigzag Level Order Traversal.
Memory Usage: 12.7 MB, less than 26.70% of C++ online submissions for Binary Tree Zigzag Level Order Traversal.
*/
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        if(!root) return {};
		vector<vector<int>> ans;
		vector<int> tmp;
		int depth = 0;
		queue<TreeNode*> q1;
		queue<TreeNode*> q2;
		q1.push(root);
		while(!q1.empty() || !q2.empty()){	//两个队列只要有一个有值就还需要遍历	
			TreeNode* cur = q1.front();						
			if(cur->left) {
				q2.push(cur->left);
			}
			if(cur->right) {
				q2.push(cur->right);
			}
			//必须先遍历后面然后出队，然后再插入到最终的返回数组中。
			q1.pop();
			tmp.push_back(cur->val);				
			if(q1.empty()){
				if(depth % 2 != 0){
					reverse(tmp.begin(), tmp.end());
				}
				ans.push_back(tmp);
				++depth;
				swap(q1, q2);
				tmp.clear();
			}						
		}
		return ans;
    }
};
```

## 代码2 双数组

```cc
/*
BFS used two vectors
Runtime: 4 ms, faster than 82.39% of C++ online submissions for Binary Tree Zigzag Level Order Traversal.
Memory Usage: 11.9 MB, less than 100.00% of C++ online submissions for Binary Tree Zigzag Level Order Traversal.
 */
class Solution1 {
 public:
	vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
		if (!root) return {};
		int flag = 0;
		vector<TreeNode*> cur, next;
		vector<vector<int>> ans;
		cur.push_back(root);
		while (!cur.empty()) {
			ans.push_back({});
			for (TreeNode* node : cur) {
				if (node->left) next.push_back(node->left);
				if (node->right) next.push_back(node->right);
			}
			if (flag) { // right to left
				int len = cur.size();
				while (len--) {
					ans.back().push_back(cur[len]->val);
				}
			} else {
				for (TreeNode* node : cur) {
					ans.back().push_back(node->val);
				}
			}
			swap(cur, next);
			flag = !flag;
			next.clear();
		}
		return ans;
	}
};
```


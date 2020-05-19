---
title: leetcode 130 Surrounded Regions
date: 2020-05-19 15:33:45
categories: 题解
tags:
- DFS
copyright: true
---

## 题意

[题目链接](<https://leetcode.com/problems/surrounded-regions/> ) 将被包围的O变成X

Given a 2D board containing `'X'` and `'O'` (**the letter O**), capture all regions surrounded by `'X'`.

A region is captured by flipping all `'O'`s into `'X'`s in that surrounded region.

**Example:**

```
X X X X
X O O X
X X O X
X O X X
```

After running your function, the board should be:

```
X X X X
X X X X
X X X X
X O X X
```

**Explanation:**

Surrounded regions shouldn’t be on the border, which means that any `'O'` on the border of the board are not flipped to `'X'`. Any `'O'` that is not on the border and it is not connected to an `'O'` on the border will be flipped to `'X'`. Two cells are connected if they are adjacent cells connected horizontally or vertically.

## DFS

记录自己没做出来的原因：对每个节点都作为入口遍历，然后在递归的时候就对原始的矩阵进行了修改，这样造成一些逻辑出现错误，然后一些特殊样例过不了。

正确做法：只从边界的O作为入口开始遍历，对活子标记，最后遍历整个矩阵，将活子变为原来的，死子变成X。然后对于矩阵这种可以访问父节点的双向图，注意增加一个标志，防止递归的时候陷入无限递归中。

另一个教训就是尽量不要在递归中去改变公有，全局变量，容易造成一些想不到的逻辑bug。

### 复杂度

时间复杂度$O(MN)$ 

### 代码

```cc
/*
Runtime: 36 ms, faster than 67.78% of C++ online submissions for Surrounded Regions.
Memory Usage: 10.3 MB, less than 100.00% of C++ online submissions for Surrounded Regions.
 */
class Solution {
 public:
	void solve(vector<vector<char>>& board) {
		int m = board.size();
		if (!m) return;
		int n = board[0].size();
		unordered_map<char, char> v{//使用<>尖角括号
			{'C', 'O'}, {'X', 'X'}, {'O', 'X'}
		};
		for (int i = 0; i < m; i++) {
			dfsCapture(board, i, 0, m, n);
			dfsCapture(board, i, n - 1, m, n);
		}
		for (int j = 0; j < n; j++) {
			dfsCapture(board, 0, j, m, n);
			dfsCapture(board, m - 1, j, m, n);
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				board[i][j] = v[board[i][j]];
			}
		}

	}
 private:
	void dfsCapture(vector<vector<char>>& board,
	                int i, int j, int m, int n) {
		if (i < 0 || i  >= m || j  < 0 || j  >= n)// 不越界
			return;
		if (board[i][j] == 'X' || board[i][j] == 'C') return; //c means no need Capture and visited;
		board[i][j] = 'C';
		dfsCapture(board, i - 1, j, m, n);
		dfsCapture(board, i, j - 1, m, n);
		dfsCapture(board, i + 1, j, m, n );
		dfsCapture(board, i, j + 1, m, n);

	}
};
```

## 总结

只能有边界的一个入口来进行遍历，入口选定好之后，用BFS也是可以的，这里忽略了。

## 参考

<https://zxi.mytechroad.com/blog/searching/leetcode-130-surrounded-regions/> 
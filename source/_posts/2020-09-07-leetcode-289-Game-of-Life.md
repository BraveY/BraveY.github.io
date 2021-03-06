---
title: leetcode 289 Game of Life
date: 2020-09-07 20:45:39
categories: 题解
tags:
- 位操作
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/game-of-life/),生命游戏的状态更新

According to the [Wikipedia's article](https://en.wikipedia.org/wiki/Conway's_Game_of_Life): "The **Game of Life**, also known simply as **Life**, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

Given a *board* with *m* by *n* cells, each cell has an initial state *live* (1) or *dead* (0). Each cell interacts with its [eight neighbors](https://en.wikipedia.org/wiki/Moore_neighborhood) (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

1. Any live cell with fewer than two live neighbors dies, as if caused by under-population.
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies, as if by over-population..
4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

Write a function to compute the next state (after one update) of the board given its current state. The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously.

**Example:**

```
Input: 
[
  [0,1,0],
  [0,0,1],
  [1,1,1],
  [0,0,0]
]
Output: 
[
  [0,0,0],
  [1,0,1],
  [0,1,1],
  [0,1,0]
]
```

**Follow up**:

1. Could you solve it in-place? Remember that the board needs to be updated at the same time: You cannot update some cells first and then use their updated values to update other cells.
2. In this question, we represent the board using a 2D array. In principle, the board is infinite, which would cause problems when the active area encroaches the border of the array. How would you address these problems?

## 方法1 替换

### 思路

根据规则，遍历一次当前状态矩阵，然后将新的状态保存到新的一个二维矩阵。最后将新状态矩阵复制回去就可以了。

### 复杂度

时间复杂度$O(MN)$ 空间复杂度$O(MN)$

### 代码

```cc
/*
Runtime: 0 ms, faster than 100.00% of C++ online submissions for Game of Life.
Memory Usage: 7.2 MB, less than 41.50% of C++ online submissions for Game of Life.
 */
class Solution1 {
  public:
	void gameOfLife(vector<vector<int>>& board) {
		int m = board.size();
		if (!m) return ;
		int n = board[0].size();
		if (!n) return;
		vector<vector<int>> state(m, vector<int>(n, 0));
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				int liveCount = 0;
				if (i - 1 >= 0 && board[i - 1][j]) ++liveCount;
				if (i - 1 >= 0 && j - 1 >= 0 && board[i - 1][j - 1]) ++liveCount;
				if (i - 1 >= 0 && j + 1 < n && board[i - 1][j + 1]) ++liveCount;
				if (j - 1 >= 0 && board[i][j - 1]) ++liveCount;
				if (j + 1  < n && board[i][j + 1]) ++liveCount;
				if (i + 1 < m && board[i + 1][j]) ++liveCount;
				if (i + 1 < m && j - 1 >= 0 && board[i + 1][j - 1]) ++liveCount;
				if (i + 1 < m && j + 1 < n && board[i + 1][j + 1]) ++liveCount;
				if (board[i][j]) { //live
					if (liveCount < 2) state[i][j] = 0;
					else if (liveCount == 2 || liveCount == 3) state[i][j] = 1;
					else state[i][j] = 0;
				} else {
					if (liveCount == 3) state[i][j] = 1;
					else state[i][j] = 0;
				}
			}
		}
		board = state;
	}

  private:
};
```

代码非常荣誉，主要在邻居状态的统计，可以用循环遍历3*3的矩阵来实现。

## 方法2 位操作

### 思路

状态只有0和1，一个int中只占用了1个bit，所以用第二个高bit来存储下一个需要变的状态。

实际上就是需要变成1的状态，使用原状态与2(二进制为10)做或运算就可以了。

### 复杂度

时间复杂度$O(MN)$ 空间复杂度$O(1)$

### 代码

```cc
/*
Runtime: 0 ms, faster than 100.00% of C++ online submissions for Game of Life.
Memory Usage: 7 MB, less than 67.48% of C++ online submissions for Game of Life.
 */
class Solution {
  public:
	void gameOfLife(vector<vector<int>>& board) {
		int m = board.size();
		if (!m) return ;
		int n = board[0].size();
		if (!n) return;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				int count = liveCount(board, i, j, m, n);
				if (count == 3 || count - board[i][j] == 3) board[i][j] |= 2; //high bit set to 1; no matter live or dead
			}
		}
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				board[i][j] >>= 1;
			}
		}
	}
  private:
	int liveCount(vector<vector<int>>& board, int i, int j, int m, int n) {
		int count = 0;
		for (int I = max(i - 1, 0); I < min(i + 2, m); ++I) {
			for (int J = max(j - 1, 0); J < min(j + 2, n); ++J) {
				count += board[I][J] & 1;
			}
		}
		return count;
	}
};
```

## 总结

1. 对于0，1这种占位很少bit的状态变量，可以考虑位操作来节省空间。

## 参考

https://leetcode.com/problems/game-of-life/discuss/73230/C%2B%2B-O(1)-space-O(mn)-time
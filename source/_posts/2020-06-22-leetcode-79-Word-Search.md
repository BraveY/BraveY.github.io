---
title: leetcode 79 Word Search
date: 2020-06-22 11:03:29
categories: 题解
tags:
- 回溯
- DFS
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/word-search/)  判断一个字符串是否在二维数组中存在

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

**Example:**

```
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```

 

**Constraints:**

- `board` and `word` consists only of lowercase and uppercase English letters.
- `1 <= board.length <= 200`
- `1 <= board[i].length <= 200`
- `1 <= word.length <= 10^3`

## 方法

### 思路

思路比较容易想得到，对每个节点的字符考虑是否与字符串的相同，如果相同就深度遍历，不同就返回，然后遍历所有可能的节点。主要是想记录下，自己之前写的代码一些小的细节没有注意，导致一个样例始终超时。

然后顺便记录下自己做回溯的题的一点小心得，个人觉得最重要的是想清楚每个节点需要遍历的子节点有哪些，一般是用for循环去寻找子节点，因为图的结构没有指针来指向子节点，如果是排列的问题的话，子节点考虑所有的，如果是组合的话设置索引不考虑之前已经遍历过的。

接着是一些状态变量的考虑，回溯最主要的思路就是，DFS遍历过一个子节点之后返回的时候，path状态变量重置回进入子节点之前的状态。

### 代码

自己之前的超时的代码

```c++
class Solution {
  public:
	bool exist(vector<vector<char>>& board, string word) {
		int m = board.size();
		int n = board[0].size();
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (word[0] == board[i][j]) {
					if (backtrack(board, word, i , j ))
						return true;
				}
			}
		}
		return false;
	}

  private:
	bool backtrack(vector<vector<char>>& board, string word, int i, int j) {
		int m = board.size();
		int n = board[0].size();
		if (!word.length()) return true;
		if ( i < 0 || i > m - 1 || j < 0 || j > n - 1) return false;
		if (word[0] != board[i][j]) return false;
		board[i][j] = '1';
		string next = word.substr(1);
		bool flag1 = backtrack(board, next, i, j - 1);
		bool flag2 = backtrack(board, next, i, j + 1);
		bool flag3 = backtrack(board, next, i - 1, j);
		bool flag4 = backtrack(board, next, i + 1, j);
		board[i][j] = word[0];
		return flag1 || flag2 || flag3 || flag4;
	}
};
```

整体逻辑是一致的，但是因为一些重复的工作导致了一个样例超时。

优化的思路

1. 不对字符串进行截取操作，而是使用状态变量d来记录当前节点应该对应那个字符
2. 不重复使用size()函数来得到边界，放在私有变量中来，避免在递归中重复调用

```cc
/*
Runtime: 40 ms, faster than 96.20% of C++ online submissions for Word Search.
Memory Usage: 11.2 MB, less than 82.77% of C++ online submissions for Word Search.
 */
class Solution {
  public:
	bool exist(vector<vector<char>>& board, string word) {
		m = board.size();
		n = board[0].size();
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (word[0] == board[i][j]) {
					if (backtrack(board, word, i , j , 0))
						return true;
				}
			}
		}
		return false;
	}

  private:
	bool backtrack(vector<vector<char>>& board, const string& word, int i, int j, int d) {
		if (word.length() == d) return true;
		if ( i < 0 || i > m - 1 || j < 0 || j > n - 1) return false;
		if (word[d] != board[i][j]) return false;
		board[i][j] = '1';
		// string next = word.substr(1);
		bool found = backtrack(board, word, i, j - 1, d + 1)
		             || backtrack(board, word, i, j + 1, d + 1)
		             || backtrack(board, word, i - 1, j, d + 1)
		             || backtrack(board, word, i + 1, j, d + 1);
		board[i][j] = word[d];
		return found;
	}
	int m , n;
};
```




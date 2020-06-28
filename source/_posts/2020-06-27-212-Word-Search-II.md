---
title: 212 Word Search II
date: 2020-06-27 23:25:01
categories: 题解
tags:
- TRIE
- 回溯
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/word-search-ii/)

Given a 2D board and a list of words from the dictionary, find all words in the board.

Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

 

**Example:**

```
Input: 
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]

Output: ["eat","oath"]
```

## 回溯

### 思路

思路比较简单就是对每个字符串都用回溯的方法判断是否可以在board中组成。也就是[79](https://bravey.github.io/2020-06-22-leetcode-79-Word-Search.html)的调用

### 代码

```c++
/*
Runtime: 1132 ms, faster than 9.24% of C++ online submissions for Word Search II.
Memory Usage: 13.6 MB, less than 93.56% of C++ online submissions for Word Search II.
 */
class Solution {
  public:
	vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
		vector<string> ans;
		for (auto str : words) {
			if (exist(board, str))
				ans.push_back(str);
		}
		return ans;
	}

  private:
	bool exist(vector<vector<char>>& board, const string& word) {
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



## TRIE前缀树

### 思路

将给定的候选字符串存储在TRIE树中，然后只进行一遍深度遍历就可以了。

记录下这个TRIE树的构成，属性上，每个节点有26个指针分别指向26个英文字母（代码上实现的时候使用一个26大小的指针数组nodes就可以了）,一个字符串指针word，记录当前节点是否是一个字符串的结尾，是的话记录该字符串的值，不是的话则为空指针。

整个的构建过程有点类似与FP树的构建。

用图来描述：

![](https://res.cloudinary.com/bravey/image/upload/v1593316256/blog/coding/trie.jpg)

拥有相同前缀的字符串会有一段相同的节点。

根据给出的候选字符串建立好TRIE树之后，与上一个回溯法正好相反，根据board里面给出的字符作为下一个节点，root做为入口去对trie树进行遍历，如果能够走到末尾的叶子节点就将叶子节点的字符串加入的ans返回数组中。

这种方法之所以比第一个方法要快是因为二者进行dfs遍历的次数不同，第一个方法每个字符串都需要对board数组进行遍历。而第二种方法只需要对board进行一次遍历就可以找出所有符合条件的。相当于避免了对有重复前缀的字符串部分进行重复遍历的过程。

### 代码

```cc
/*
Runtime: 164 ms, faster than 54.99% of C++ online submissions for Word Search II.
Memory Usage: 36.4 MB, less than 51.95% of C++ online submissions for Word Search II.
 */
class TrieNode {
public:
  vector<TrieNode*> nodes;  //指向下一个节点，有26个可以选择
  const string* word; //对应的字符串
  TrieNode(): nodes(26), word(nullptr) {}
  ~TrieNode() {
    for (auto node : nodes) delete node;
  }  
};
class Solution {
public:
  vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
    TrieNode root;
    
    // Add all the words into Trie.
    for (const string& word : words) {
      TrieNode* cur = &root; // 每次从root开始构建
      for (char c : word) {
        auto& next = cur->nodes[c - 'a'];//字母的索引 从a开始
        if (!next) next = new TrieNode(); //没有已经建立的字符节点则新建
        cur = next;
      }
      cur->word = &word;
    }
    
    const int n = board.size();
    const int m = board[0].size();    
    vector<string> ans;
    
    function<void(int, int, TrieNode*)> walk = [&](int x, int y, TrieNode* node) { //回溯的过程     
      if (x < 0 || x == m || y < 0 || y == n || board[y][x] == '#')
        return;      
      
      const char cur = board[y][x];
      TrieNode* next_node = node->nodes[cur - 'a'];
      
      // Pruning, only expend paths that are in the trie.
      if (!next_node) return;
      
      if (next_node->word) {
        ans.push_back(*next_node->word);
        next_node->word = nullptr;
      }
 
      board[y][x] = '#';
      walk(x + 1, y, next_node);
      walk(x - 1, y, next_node);
      walk(x, y + 1, next_node);
      walk(x, y - 1, next_node);
      board[y][x] = cur;
    };
    
    // Try all possible pathes.
    for (int i = 0 ; i < n; i++)
      for (int j = 0 ; j < m; j++)
        walk(j, i, &root);        
        
    return ans;
  }
};
```

## 参考

https://zxi.mytechroad.com/blog/searching/leetcode-212-word-search-ii/
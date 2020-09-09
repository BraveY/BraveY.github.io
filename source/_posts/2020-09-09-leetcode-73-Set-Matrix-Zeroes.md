---
title: leetcode 73 Set Matrix Zeroes
date: 2020-09-09 13:06:00
categories: 题解
tags:
- 二维数组
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/set-matrix-zeroes/) 将为0的元素所在的行和列都置为0.

Given an `*m* x *n*` matrix. If an element is **0**, set its entire row and column to **0**. Do it [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm).

**Follow up:**

- A straight forward solution using O(*m**n*) space is probably a bad idea.
- A simple improvement uses O(*m* + *n*) space, but still not the best solution.
- Could you devise a constant space solution?

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg)

```
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/08/17/mat2.jpg)

```
Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
```

 

**Constraints:**

- `m == matrix.length`
- `n == matrix[0].length`
- `1 <= m, n <= 200`
- `-231 <= matrix[i][j] <= 231 - 1`

## 方法1 替换

### 思路

为了避免被动置为0的元素的影响，依然是建立个完整的替换数组来存储下一阶段的状态从而不会造成遍历时的冲突。

### 复杂度

时间复杂度为$O(MN)$ 

空间复杂度 $O(MN)$

### 代码

```cc
/*
Runtime: 24 ms, faster than 94.45% of C++ online submissions for Set Matrix Zeroes.
Memory Usage: 13.7 MB, less than 5.26% of C++ online submissions for Set Matrix Zeroes.
space:O(m*n)
 */
class Solution1 {
  public:
	void setZeroes(vector<vector<int>>& matrix) {
		int m = matrix.size();
		int n = matrix[0].size();
		vector<vector<int>> dummy(m, vector<int>(n, 1));
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (matrix[i][j] == 0) {
					for (int k = 0; k < m; ++k) {
						dummy[k][j] = 0;
					}
					for (int k = 0; k < n; ++k) {
						dummy[i][k] = 0;
					}
				} else {
					if (dummy[i][j] != 0) dummy[i][j] = matrix[i][j];
				}
			}
		}
		matrix = dummy;
	}

  private:
};
```

## 方法2 

### 思路

方法1 记录了所有的二维数组的状态，实际上只需要用一个col数组和row数组来记录哪些行与列需要置为0就可以了。

### 复杂度

时间复杂度为$O(MN)$ 

空间复杂度 $O(M+N)$

### 代码

```cc
/*
Runtime: 24 ms, faster than 94.45% of C++ online submissions for Set Matrix Zeroes.
Memory Usage: 13.3 MB, less than 53.91% of C++ online submissions for Set Matrix Zeroes.
space:O(m + n)
 */
class Solution2 {
  public:
	void setZeroes(vector<vector<int>>& matrix) {
		int m = matrix.size();
		int n = matrix[0].size();
		vector<int> row;
		vector<int> col;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (matrix[i][j] == 0) {
					row.push_back(i);
					col.push_back(j);
				}
			}
		}
		for (int i = 0; i < row.size(); ++i) {
			for (int j = 0; j < n; ++j) {
				matrix[row[i]][j] = 0;
			}
		}
		for (int i = 0; i < col.size(); ++i) {
			for (int j = 0; j < m; ++j) {
				matrix[j][col[i]] = 0;
			}
		}
	}

  private:
};
```

## 方法3

### 思路

同方法2一样，不过将记录的状态的两个数组分别直接用二维数组的首行和首列来记录。记录方法是将0元素对应的首行与首列元素置为0 ，表示本行或者本列需要置为0 。为了不影响首行与首列的变化，需要在遍历的时候记录，是否原来就有0元素。原来就有的话，对应的首行与首列都需要置为0，否则不需要变化原来的元素。

在第二次修改状态的时候，需要把首行与首列排除在外。

### 复杂度

时间复杂度为$O(MN)$ 

空间复杂度 $O(1)$

### 代码

```cc
/*
Runtime: 28 ms, faster than 77.73% of C++ online submissions for Set Matrix Zeroes.
Memory Usage: 13.2 MB, less than 71.57% of C++ online submissions for Set Matrix Zeroes.
 */
class Solution {
  public:
	void setZeroes(vector<vector<int>>& matrix) {
		int m = matrix.size();
		int n = matrix[0].size();
		bool row0 = false;
		bool col0 = false;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				if (matrix[i][j] == 0) {
					if (i == 0) row0 = true;
					if (j == 0) col0 = true;
					matrix[i][0] = 0;
					matrix[0][j] = 0;
				}
			}
		}
		for (int i = 1; i < m; ++i) {
			for (int j = 1; j < n; ++j) {
				if (matrix[0][j] == 0 || matrix[i][0] == 0)
					matrix[i][j] = 0;
			}
		}
		if (row0) {
			for (int i = 0; i < n; ++i) {
				matrix[0][i] = 0;
			}
		}
		if (col0) {
			for (int i = 0; i < m; ++i) {
				matrix[i][0] = 0;
			}
		}
	}

  private:
};
```

## 参考

https://github.com/azl397985856/leetcode/blob/master/problems/73.set-matrix-zeroes.md
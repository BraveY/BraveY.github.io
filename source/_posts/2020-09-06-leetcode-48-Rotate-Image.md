---
title: leetcode 48 Rotate Image
date: 2020-09-06 16:41:56
categories: 题解
tags:
- 数组
copyright: true

---

## 题意

[题目链接](https://leetcode.com/problems/rotate-image/), 将二维矩阵顺时针90度旋转。

You are given an *n* x *n* 2D `matrix` representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg)

```
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2020/08/28/mat2.jpg)

```
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```

**Example 3:**

```
Input: matrix = [[1]]
Output: [[1]]
```

**Example 4:**

```
Input: matrix = [[1,2],[3,4]]
Output: [[3,1],[4,2]]
```

 

**Constraints:**

- `matrix.length == n`
- `matrix[i].length == n`
- `1 <= n <= 20`
- `-1000 <= matrix[i][j] <= 1000`

## 方法1 镜像

### 思路

先沿着对角线镜像，然后沿着y轴镜像。

### 复杂度

时间复杂度$O(N^2)$, 空间复杂度$O(1)$

### 代码

来自花花

```cc
/*
Runtime: 0 ms, faster than 100.00% of C++ online submissions for Rotate Image.
Memory Usage: 7.5 MB, less than 5.14% of C++ online submissions for Rotate Image.
 */
class Solution {
  public:
	void rotate(vector<vector<int>>& matrix) {
		const int n = matrix.size();
		for (int i = 0; i < n; ++i)
			for (int j = i + 1; j < n; ++j)
				swap(matrix[i][j], matrix[j][i]); //First pass: mirror around diagonal
		for (int i = 0; i < n; ++i)
			reverse(begin(matrix[i]), end(matrix[i])); //Second pass: mirror around y axis
	}
};
```

## 方法2 队列

自己的思路，写起来很繁琐，不推荐。

### 思路

按顺时针的方向，将每一圈的数字，存储起来，然后重新按序排列。

### 复杂度

时间复杂度$O(N^2)$, 空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 0 ms, faster than 100.00% of C++ online submissions for Rotate Image.
Memory Usage: 7.2 MB, less than 56.48% of C++ online submissions for Rotate Image.
 */
class Solution1 {
  public:
	void rotate(vector<vector<int>>& matrix) {
		int m = matrix.size();
		if (!m) return;
		int n = matrix[0].size();
		if (!n || m != n) return;
		int count = m;
		queue<int> seq;
		int i = 0;
		while (count > 0) { //(i,i) is the corner start
			int j = i;
			// cout << "count: " << count <<  endl;
			// cout << "i: " << i <<  endl;
			for (int k = i; k < i + count - 1; ++k) {
				seq.push(matrix[j][k]);
			}
			// cout << "1" << endl;
			j = i + count - 1;
			for (int k = i; k < i + count - 1; ++k) {
				seq.push(matrix[k][j]);
			}
			// cout << "2" << endl;
			for (int k = i + count - 1; k > i; --k) {
				seq.push(matrix[j][k]);
			}
			// cout << "3" << endl;
			j = i;
			for (int k = i + count - 1; k > i ; --k) {
				seq.push(matrix[k][j]);
			}
			// cout << "4" << endl;
			j = i + count - 1;
			for (int k = i; k < i + count - 1; ++k) {
				matrix[k][j] = seq.front();
				seq.pop();
			}
			// cout << "-1" << endl;
			for (int k = i + count - 1; k > i; --k) {
				matrix[j][k] = seq.front();
				seq.pop();
			}
			// cout << "-2" << endl;
			j = i;
			for (int k = i + count - 1; k > i ; --k) {
				matrix[k][j] = seq.front();
				seq.pop();
			}
			// cout << "-3" << endl;
			for (int k = i; k < i + count - 1; ++k) {
				matrix[j][k] = seq.front();
				seq.pop();
			}
			// cout << "-4" << endl;
			count -= 2;
			++i;
		}
	}

  private:
};

```



## 总结

1. 不容易想出来做两次镜像，所以记录下
2. 其他变种包括旋转180度，270度等。

## 参考

[花花leetcode](https://zxi.mytechroad.com/blog/algorithms/array/leetcode-48-rotate-image/)
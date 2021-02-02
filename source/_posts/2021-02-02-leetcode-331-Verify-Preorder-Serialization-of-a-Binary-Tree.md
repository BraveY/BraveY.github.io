---
title: leetcode 331 Verify Preorder Serialization of a Binary Tree
date: 2021-02-02 21:23:14
categories: 题解
tags:
- DFS
- 递归
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/)，判断一个字符串序列是否为有效的二叉树先序遍历序列。

One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as `#`.

```
     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \
# # # #   # #
```

For example, the above binary tree can be serialized to the string `"9,3,4,#,#,1,#,#,2,#,6,#,#"`, where `#` represents a null node.

Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.

Each comma separated value in the string must be either an integer or a character `'#'` representing `null` pointer.

You may assume that the input format is always valid, for example it could never contain two consecutive commas such as `"1,,3"`.

**Example 1:**

```
Input: "9,3,4,#,#,1,#,#,2,#,6,#,#"
Output: true
```

**Example 2:**

```
Input: "1,#"
Output: false
```

**Example 3:**

```
Input: "9,#,#,1"
Output: false
```

## 方法1 递归

### 思路

如果是数字则说明为是非空节点，递归判断其左右节点是否有效。如果不是数字则判断是否为#号。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 4 ms, faster than 81.46% of C++ online submissions for Verify Preorder Serialization of a Binary Tree.
Memory Usage: 6.8 MB, less than 87.54% of C++ online submissions for Verify Preorder Serialization of a Binary Tree.
*/
class Solution {
  public:
    bool isValidSerialization(string preorder) {
        int pos = 0;
        return isValid(preorder, pos) && pos == preorder.size();
    }

  private:
    bool isValid(const string& s, int& pos){
        if(pos >= s.size()) return false;
        if(isdigit(s[pos])){
            while(isdigit(s[pos])) ++pos;
            return isValid(s, ++pos) && isValid(s, ++pos);
        }
        return s[pos++] == '#'; // 递归的返回遇到非数字节点
    }
};
```

## 总结

1. 代码上使用int&引用来代替全局变量
2. 对于null节点还有值的无效序列，其在递归的时候提前终止，pos值不会等于序列长度
3. 使用`isdigit(char c)`来判断是否为数字，比较常见的字符串操作。

## 参考

[huahua](https://zxi.mytechroad.com/blog/tree/leetcode-331-verify-preorder-serialization-of-a-binary-tree/)
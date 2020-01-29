---
title: leetcode 543 Diameter of Binary Tree
date: 2020-01-06 19:08:33
categories: 题解
tags:
- 分治
- 二叉树
copyright: true
---

# leetcode 543 Diameter of Binary Tree 

[题目来源](<https://leetcode.com/problems/diameter-of-binary-tree/> )。需要求解一颗二叉树的最大路径的长度，第一次做出来了，第二次没有做对，所以记录下。

<!--more-->

## 思路

首先观察到问题的输入是一颗二叉树，其左右子节点仍然是一个二叉树，属于可以分解成子问题的数据结构，不是求解最优解，所以很自然的想到分治的方法。

对于每一个节点而言，经过该节点的最长路径分为两种情况：

1. 不经过其父节点，是左节点的最长深度+右节点的最长深度
2. 经过其父节点，且经过其左右子节点中拥有最大深度的一个子节点

第二种情况对于该节点的父节点而言，是第一种情况，因此对于输入的二叉树而言，更新最大深度只存在第一种情况就是某一节点的左节点最大深度与右节点的最大深度相加，因此每个节点需要返回其左右子节点中拥有最大深度的一个子节点。这样从得到叶子节点这个最小的子问题的解开始，就可以不断向上叠加，得到每一个节点的解，最终得到原始问题的解了。

所以这里的子问题应该是**左右子节点的最长深度**，而不是左右子节点的最长路径，因为每个节点只能经过一次，也就是对于父节点而言，**只能选择左边或者右边的一条路径**，不能左右两边都同时选择。我第二次做错就是因为理解成子问题是每个子节点对应的最大路径长度了。

用分治的三个过程来表示为：

- **划分**：将输入的二叉树划分为左子树L，与右子树R，两个子树仍然是二叉树。
- **治理**：将L和R分别递归调用函数求得两个子树的最大深度。
- **合并**：将左子树与右子树的最大深度相加来更新当前的最大路径值，之后返回该子树两个子节点中最大深度最大的深度值。

因为子问题**没有直接返回最大路径长度**，所以需要选择一个全局变量来存储递归返回时的最大路径长度。同时需要注意让**NULL节点返回-1**而不是0，计算长度的时候通过+1来表示路径长度。

## 复杂度

用T(n)表示对于一个拥有n个节点的二叉树寻找最大路径长度所需要的时间，T(n)最复杂的情况是二叉树是一棵完全二叉树。对最复杂的情况进行分析有$T(n)=2T(\lfloor\frac{n}{2}\rfloor)+C$
$$
\begin{equation}\nonumber
\begin{split}
T(n)&=2T[{\lfloor\frac{n}{2}\rfloor]+C}\\
&=2^2T[{\lfloor\frac{n}{2^2}\rfloor]+C+2C}\\
&=2^3T[{\lfloor\frac{n}{2^3}\rfloor]+C+2C+2C}\\
\dots\\
&=2^kT[{\lfloor\frac{n}{2^k}\rfloor]+(2k-1)C}\\
\end{split} 
\end{equation}
$$
每次分解后规模变为左右两个子树，因此下降到$\lfloor\frac{n}{2}\rfloor$,而每次比较左右两个子树最大深度与更新最大路径长度所需的时间为常数C。设当第k次的时候下降到规模为1的叶子节点，因此有$\frac{n}{2^k}=1$推出$k=\log_2n$所以有$T(n)=n+(2\log_2n-1)C=O(n)$,所以时间复杂度为O(n)。

## 代码

代码如下，[源码链接](<https://github.com/BraveY/Coding/blob/master/leetcode/diameter-of-binary-tree.cc> )。

```cc
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int ans = 0;
    int diameterOfBinaryTree(TreeNode* root) {
        LP(root);
        return ans;
    }
private:
    int LP(TreeNode* root){
        if(!root) return -1;
        //求左节点的最大深度与右节点的最大深度，根节点到子节点还有长度1，所以都需要加上1
        int l = LP(root->left)+1;
        int r = LP(root->right)+1;
        ans = max(ans, l+r);
        return max(l,r);
    }
};
```


---
title: leetcode 287 Find the Duplicate Number
date: 2020-04-16 22:43:49
categories: 题解
tags:
- 二分搜索
- 链表
copyright: true
---

# leetcode 287 Find the Duplicate Number

## 题目描述

[题目链接](<https://leetcode.com/problems/find-the-duplicate-number/> ),寻找一个数组中的重复值。

Given an array *nums* containing *n* + 1 integers where each integer is between 1 and *n* (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

**Example 1:**

```
Input: [1,3,4,2,2]
Output: 2
```

**Example 2:**

```
Input: [3,1,3,4,2]
Output: 3
```

**Note:**

1. You **must not** modify the array (assume the array is read only).
2. You must use only constant, *O*(1) extra space.
3. Your runtime complexity should be less than $O(n^2)$.
4. There is only one duplicate number in the array, but it could be repeated more than once.

## 思路

### 二分搜索

题目给出了数组中对数字大小的限制，从这点出发，假设数组没有重复，并且按照顺序排序的话，则可以从1排到n+1。但是因为题目要求最多排到n，所以最后一个n+1一定要换成k，而$1\le k\le n$所以一定会与前n个值产生重复。

当产生重复以后从k到n的元素有：小于等于其值的个数一定会大于其本身，即满足$A[i] \le A[k]$ 的i的个数有$count(A[i])\ge k$ 。这样只用寻找到满足前面条件的最小的i,就找到重复值了。

![](https://res.cloudinary.com/bravey/image/upload/v1587131587/blog/coding/lc287_1.jpg)

使用二分搜索来进行查找，如果小于中间值的个数等于中间值，说明左半部分没有重复值，向右边寻找。否则向左边寻找，直到找到最小值。

### 链表找环

索引从0开始，得到A[0]的值，然后将A[0]的值作为新的索引去寻找一个索引，按照这样的规则形成一个链表，因为一定有重复的值，相当于同一个索引会被指向两次，因此会形成环。过程如下所示： 

![](https://res.cloudinary.com/bravey/image/upload/v1587131587/blog/coding/lc287_2.jpg)

链表中找环，可以使用快慢指针的方法。

快慢指针同时从0索引开始移动，慢指针每次移动一步，快指针每次移动两步，这样快指针就会先进入环中，然后一直在环里面移动，直到慢指针也进入环中，然后二者第一次相遇。

当两个指针相遇后，让快指针停止移动，然后新用一个指针从头开始移动。慢指针和新指针都是一次移动一步，当慢指针和新指针再次相遇时，到达环的入口也就是，重复的值。

### 数学解释

![](https://res.cloudinary.com/bravey/image/upload/v1587131587/blog/coding/lc287_3.jpg)

设环前面的长度为a，快慢指针的相遇点距离环的起点距离为b，剩下的距离为c。

则快慢指针相遇时，慢指针走的路程是：$a+b$, 快指针走的路程是$a+k(b+c)+b$ ，$k(b+c)$ 指的是快指针已经在环中走过k圈。因为快指针的速度是慢指针的两倍，所以快指针走过的路程是慢指针的两倍，因此有等式$2(a+b)=a+(k-1)(b+c)+b$ 化简后得到$a = (k-1)b + kc$ 因此这个时候再走a的距离就会回到起点。

所以新指针与慢指针会相遇到环的起点。

## 复杂度

### 二分搜索

二分搜索，树的高度是$O(log(n))$ ，每次统计count需要$O(n)$,因此总共的时间复杂度是$O(nlog(n))$ 

### 链表找环

使用两个循环，因此复杂度是$O(n)$.

##　代码

[完整代码地址](https://github.com/BraveY/Coding/blob/master/leetcode/287find-the-duplicate-number.cc)

### 二分搜索

```cc
/*
Runtime: 12 ms, faster than 60.68% of C++ online submissions for Find the Duplicate Number.
Memory Usage: 7.6 MB, less than 100.00% of C++ online submissions for Find the Duplicate Number.
 */
class Solution1 {
  public:
    int findDuplicate(vector<int>& nums) {
        int l = 1;
        int r = nums.size();
        while (l < r) {
            int m = (r - l) / 2 + l;
            int count = 0; // len(nums <= m)
            for (int num : nums)
                if (num <= m) ++count;
            if (count <= m)
                l = m + 1;
            else
                r = m;
        }
        return l;
    }
};
```



### 链表找环

```cc
class Solution3 {
  public:
    int findDuplicate(vector<int>& nums) {
        //最开始的index是从0开始的
        int slow = nums[0];
        int fast = nums[0];
        while (true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow == fast) break;
        }
        int find = 0;
        while (find != slow) {
            find = nums[find];
            slow = nums[slow];
        }
        return find;
    }
};
```

## 参考

[leetcode算法汇总 （三）快慢指针 - mercury的文章 - 知乎]( https://zhuanlan.zhihu.com/p/72886883 )

[花花leetcode](<https://zxi.mytechroad.com/blog/algorithms/binary-search/leetcode-287-find-the-duplicate-number/> )
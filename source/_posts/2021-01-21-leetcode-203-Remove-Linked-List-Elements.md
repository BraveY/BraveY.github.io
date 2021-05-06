---
title: leetcode 203 Remove Linked List Elements
date: 2021-01-21 22:21:54
categories: 题解
tags:
- 链表
- 指针
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/remove-linked-list-elements/)

Remove all elements from a linked list of integers that have value ***val***.

**Example:**

```
Input:  1->2->6->3->4->5->6, val = 6
Output: 1->2->3->4->5
```

## 方法1

### 思路

判断当前节点的下一个节点的值是否需要删除，是的话直接将next指针指向被删除的下一个节点。

思路很简单主要是第一次写有很多小bug没有debug出来，记下方便后面总结。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

删除节点之后不要继续保持指针不变，防止[1,1]  1 这样连续删除得case 遗漏了对下一个得判断

```cc
/*
Runtime: 24 ms, faster than 92.80% of C++ online submissions for Remove Linked List Elements.
Memory Usage: 14.8 MB, less than 100.00% of C++ online submissions for Remove Linked List Elements.
*/
class Solution {
  public:
    ListNode* removeElements(ListNode* head, int val) {
        if (!head) return nullptr;
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* cur = dummy; // 在新链表之前插入一个新节点，方便操作需要删除得节点就是首节点得问题
        while(cur) { //自己最开始写得是!cur 记住加上！表示进入循环得是空指针，这里显然是需要非空指针
            if (cur->next && cur->next->val == val) { //用到指针得地方就先判断是否为非空 自己最开始没有加入cur->next 是否为空指针得判断
                cur->next = cur->next->next;
            } else cur = cur->next; //最开始没有加入else 而是直接放入if 语句后面 这样无法通过[1,1] 1这个例子
        }
        return dummy->next;
    }

  private:
};
```

## 总结

一些代码上的小漏洞，不考虑清楚会很容易浪费时间。详细看代码注释。

循环每次只做一件事：要么移动next指针，要么移动cur指针，不加else就变成移动next指针后再移动cur指针，不够分离。

## 参考
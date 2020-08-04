---
title: leetcode 19 Remove Nth Node From End of List
date: 2020-08-04 11:10:56
categories: 题解
tags: 
- 双指针
- 链表
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) 删除指定节点

Given a linked list, remove the *n*-th node from the end of list and return its head.

**Example:**

```
Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
```

**Note:**

Given *n* will always be valid.

**Follow up:**

Could you do this in one pass?

## 方法1 数组

### 思路

遍历一次链表然后把每个位置的节点地址记录在数组中，之后执行删除操作。

### 复杂度

时间复杂度和空间复杂度都是$O(N)$

### 代码

```cc
/*
Runtime: 4 ms, faster than 83.01% of C++ online submissions for Remove Nth Node From End of List.
Memory Usage: 10.7 MB, less than 5.27% of C++ online submissions for Remove Nth Node From End of List.
 */
class Solution1 {
  public:
	ListNode* removeNthFromEnd(ListNode* head, int n) {
		ListNode* cur = head;
		vector<ListNode*> nodes;
		while (cur) {
			nodes.push_back(cur);
			cur = cur->next;
		}
		int len = nodes.size();
		if (len == 1) return NULL;
		int pos = len - n ;
		if (pos == len - 1) nodes[pos - 1] ->next = NULL;
		else if (pos == 0) return nodes[pos + 1];
		else {
			nodes[pos - 1]->next = nodes[pos + 1];
		}
		return head;
	}

  private:
};
```

## 方法2 双指针

### 思路

在头指针之前加入一个新的头指针，然后fast指针先走指定的n步，之后快慢指针同时移动直到快指针到达结尾。此时慢指针的下一个节点就是需要被删除的节点。

可以理解成先形成指定长度的移动窗口或者尺子，然后移动尺子这样来定位需要删除的节点

### 复杂度

时间复杂度是$O(N)$， 空间复杂度$O(1)$

### 代码

```cc
/*
Runtime: 4 ms, faster than 83.01% of C++ online submissions for Remove Nth Node From End of List.
Memory Usage: 10.4 MB, less than 72.66% of C++ online submissions for Remove Nth Node From End of List.
 */
class Solution {
  public:
	ListNode* removeNthFromEnd(ListNode* head, int n) {
		if (!head) return nullptr;
		ListNode newHead(-1);
		newHead.next = head;
		ListNode* fast = &newHead;
		ListNode* slow = &newHead;
		for (int i = 0; i < n; ++i) {
			fast = fast->next;
		}
		while (fast->next) { //需要指向待删除节点的上一个
			fast = fast->next;
			slow = slow->next;
		}
		ListNode* toDelete =  slow->next;
		slow->next = slow->next->next;
		delete toDelete;//传入指针 防止内存泄漏
		return newHead.next;
	}

  private:
};
```

## 总结

1. 防止内存泄漏
2. 尺子/窗口的思想

## 参考

[discuss](https://leetcode.com/problems/remove-nth-node-from-end-of-list/discuss/?currentPage=1&orderBy=most_votes&query=)


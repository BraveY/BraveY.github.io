---
title: leetcode 148 Sort List
date: 2020-04-29 16:20:16
categories: 题解
tags:
- 链表
- 排序
copyright: true
---

## 题意

[题目链接](<https://leetcode.com/problems/sort-list/> ) 对链表进行排序，常数空间复杂度和$O(nlogn)$ 的时间复杂度

Sort a linked list in *O*(*n* log *n*) time using constant space complexity.

**Example 1:**

```
Input: 4->2->1->3
Output: 1->2->3->4
```

**Example 2:**

```
Input: -1->5->3->4->0
Output: -1->0->3->4->5
```

## 方法

排序能够$O(nlogn)$ 的有三种，堆排序，快排，归并排序。快排是需要两个指针同时向中间移动的，题目是单链表所以排除了。

自己之前没有接触过用链表的排序，所以没想到怎么对链表进行一分为二来分治。实际上需要使用快慢指针来进行。

### 归并排序 递归

用快慢指针来寻找中间的节点，快指针每次走两步，慢指针每次走一步，当快指针到达末尾的时候，慢指针走了一半，因此也就是中间节点。后续的递归调用方法和之前的用数组的形式，大体一致，参考之前的[逆序对计数](https://bravey.github.io/2019-10-07-%E9%80%86%E5%BA%8F%E5%AF%B9%E7%9A%84%E8%AE%A1%E6%95%B0.html)。

### 复杂度

时间复杂度是$O(nlogn)$ 但是空间复杂度是$O(logn)$ ,其实有点疑惑递归的怎么算空间复杂度是根据递归的调用次数来吗？

### 代码

```cc
/*
Runtime: 40 ms, faster than 100.00% of C++ online submissions for Sort List.
Memory Usage: 12.8 MB, less than 25.00% of C++ online submissions for Sort List.
 */
class Solution {
 public:
	ListNode* sortList(ListNode* head) {
		if (!head || !head->next) return head; //递归必备的出口
		ListNode* slow = head;
		ListNode*	fast = head->next;
		// 寻找中间的节点
		while (fast && fast->next) {
			//fast 刚好到达末尾的时候 fast->next 为nullptr
			//或者刚好超过末尾1个，fast为nullptr
			slow = slow->next;
			fast = fast->next->next;
		}
		ListNode* l2 = slow->next;
		slow->next = NULL;// 中间断开
		head = sortList(head);
		l2 = sortList(l2);
		head = merge(head, l2);//merge到前半部分上面
		return head;
	}

 private:
	ListNode* merge(ListNode* l1, ListNode* l2) {
		ListNode dummy(0);
		ListNode* tail = &dummy;
		while (l1 && l2) {
		if (l1->val > l2->val) swap(l1, l2);
			// swap 交换的是变量值，对链表而言，交换了整个链表。
			//始终让l1指向当前最小的头节点的链表，简洁些，不用if else
			tail->next = l1;
			l1 = l1->next;
			tail = tail->next;
		}
		tail->next = l1 ? l1 : l2;
		return dummy.next;
	}
};	
```

需要注意下的是swap的使用，`if (l1->val > l2->val) swap(l1, l2); ` 这一句swap交换的不是节点的值，而是整个链表。

### 归并排序 循环

为了实现$O(1)$的空间复杂度，需要将递归改成循环，也就是自己实现递归的逻辑。思路是从最底层n=1的时候开始，两两合并，然后是n=2，4...这样一直到最后完成合并。

所以需要一个专门的函数来实现按照长度n对链表分组，然后在循环里面迭代调用完成分组。另外merge也需要返回头和尾，来方便相邻之间分组的连接。

### 复杂度

时间复杂度是$O(nlogn)$ 但是空间复杂度是$O(1)$ 

### 代码

代码其实有点难度的，特别是主循环中的逻辑，感觉自己下次不一定能够复现出来。

```cc
class Solution2 {
 public:
	ListNode* sortList(ListNode* head) {
		// 0 or 1 element, we are done.
		if (!head || !head->next) return head;

		int len = 1;
		ListNode* cur = head;
		while (cur = cur->next) ++len;

		ListNode dummy(0);
		dummy.next = head;
		ListNode* l;
		ListNode* r;
		ListNode* tail;
		for (int n = 1; n < len; n <<= 1) {
			cur = dummy.next; // partial sorted head
			tail = &dummy;
			while (cur) {
				l = cur;
				r = split(l, n);
				cur = split(r, n);
				auto merged = merge(l, r);
				tail->next = merged.first;  // 合并list的head 让前面合并后末尾指向新合并的头
				tail = merged.second; // 合并list的tail ，然后重新指向新合并的末尾
			}
		}
		return dummy.next;
	}
 private:
	// Splits the list into two parts, first n element and the rest.
	// Returns the head of the rest.
	ListNode* split(ListNode* head, int n) {
		while (--n && head) // 分割的长度可能超过head 的长度
			head = head->next;
		ListNode* rest = head ? head->next : nullptr;
		if (head) head->next = nullptr; // 如果有连接 就断开
		return rest;
	}

	// Merges two lists, returns the head and tail of the merged list.
	pair<ListNode*, ListNode*> merge(ListNode* l1, ListNode* l2) {
		ListNode dummy(0);
		ListNode* tail = &dummy;
		while (l1 && l2) {
			if (l1->val > l2->val) swap(l1, l2);
			tail->next = l1;
			l1 = l1->next;
			tail = tail->next;
		}
		tail->next = l1 ? l1 : l2; // 剩下的连上
		while (tail->next) tail = tail->next; //找到末尾， 方便后续的接
		return {dummy.next, tail};
	}
};
```

```cc
		tail->next = l1 ? l1 : l2; // 剩下的连上
		while (tail->next) tail = tail->next; //找到末尾， 方便后续的接
```

这里需要稍微说明的是taila->next 的修改就是对原来的节点next指针进行修改，后tail重新只想其他节点后，前面的修改依然会保留，而不是重置了。

## 参考

[huahua](<https://www.bilibili.com/video/BV1jW411d7z7?t=1100> ) 代码值得学习
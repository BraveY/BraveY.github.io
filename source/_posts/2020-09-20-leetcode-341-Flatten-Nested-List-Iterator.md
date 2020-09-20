---
title: leetcode 341 Flatten Nested List Iterator
date: 2020-09-20 21:29:18
categories: 题解
tags:
- 栈
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/flatten-nested-list-iterator/) 按顺序打印出嵌套结构的列表。

Given a nested list of integers, implement an iterator to flatten it.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

**Example 1:**

```
Input: [[1,1],2,[1,1]]
Output: [1,1,2,1,1]
Explanation: By calling next repeatedly until hasNext returns false, 
             the order of elements returned by next should be: [1,1,2,1,1].
```

**Example 2:**

```
Input: [1,[4,[6]]]
Output: [1,4,6]
Explanation: By calling next repeatedly until hasNext returns false, 
             the order of elements returned by next should be: [1,4,6].
```

## 方法1 栈

### 思路

这种有嵌套规则的很容易想到使用栈来进行匹配。

自己最开始卡壳在想在$next()$中对是嵌套list进行递归处理，然后发现出现$[[[]]]$ 这样的空列表在$next()$函数中无法返回具体值，从而出现了函数必须返回值的矛盾。

实际上$next()$函数只处理是整数值的情况，这样一定可以有返回值。为了能够一定使用$next()$返回整数值，则需要在栈中存放只包含整数值的嵌套列表，因此需要在$hashNext()$函数中将嵌套列表使用循环展开，然后依次存入，当栈顶是整数值的时候就可以暂时不用拆开了，因为可以保证$next()$一定会有值可以输出。

为了能够从栈顶输出值，则需要先在初始化函数中逆序的把嵌套列表的元素依次入栈。

### 复杂度

时间复杂度$O(N)$ 所有元素只会输出一次。

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 28 ms, faster than 53.28% of C++ online submissions for Flatten Nested List Iterator.
Memory Usage: 14.9 MB, less than 20.17% of C++ online submissions for Flatten Nested List Iterator.
 */
class NestedIterator {
  public:
	NestedIterator(vector<NestedInteger> &nestedList) {
		for (int i = nestedList.size() - 1; i >= 0; --i) {
			s.push(nestedList[i]);
		}
	}

	int next() {
		int ans = s.top().getInteger();
		s.pop();
		return ans;
	}

	bool hasNext() {
		while (!s.empty()) {
			NestedInteger temp = s.top();
			if (temp.isInteger()) return true;
			else {
				s.pop();
				vector<NestedInteger> tmpList = temp.getList();
				for (int i = tmpList.size() - 1; i >= 0; --i) {
					s.push(tmpList[i]);
				}
			}
		}
		return false;
	}

  private:
	stack<NestedInteger> s;
};
```

## 总结

如果自己想的时候情况太复杂，一般就是思路有问题了，需要重新思考。

## 参考

https://leetcode.com/problems/flatten-nested-list-iterator/discuss/80345/Share-my-C%2B%2B-solutionseasy-to-understand
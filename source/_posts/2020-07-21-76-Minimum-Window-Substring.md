---
title: 76 Minimum Window Substring
date: 2020-07-21 17:29:04
categories: 题解
tags:
- 哈希表
- 滑动窗口
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/minimum-window-substring/)找含有指定字符的最小字符串

Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

**Example:**

```
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
```

**Note:**

- If there is no such window in S that covers all characters in T, return the empty string `""`.
- If there is such window, you are guaranteed that there will always be only one unique minimum window in S.

## 方法1

自己的做法

### 思路

利用滑动窗口的思路，判断窗口是否符合条件的方法是设置两个128长的数组作为哈希表，哈希表存储每个字符出现的次数。一个哈希表记录窗口的字符次数情况，另一个记录条件字符串中的字符出现次数。只有当前窗口的对应目标字符串的字符的出现次数大于条件字符串中的次数，才表明含有条件字符串的字符。

之后就是一样滑动窗口的框架了：如果没有包含条件字符串的字符则右边窗口向右增加。如果满足了条件，则左边窗口向右滑动减少窗口长度，知道窗口刚好不满足条件。然后利用一个常量来记录出最小的满足条件的字符串窗口长度坐标，最终返回这个字符串。

### 复杂度

考虑到判断是否包含条件字符串需要使用一个循环来对两个数组哈希表进行遍历判断，这个的时间复杂度为$O(128)$。然后考虑最坏的情况，认为s的每个字符都需要两次判断（作为窗口的左边与右边各需要一次）。所以总的时间复杂度可以认为是$O(256N)$

空间复杂度为$O(256)$

### 代码

```cc
/*
Runtime: 100 ms, faster than 25.03% of C++ online submissions for Minimum Window Substring.
Memory Usage: 8 MB, less than 39.26% of C++ online submissions for Minimum Window Substring.
 */
class Solution2 {
  public:
	string minWindow(string s, string t) {
		vector<int> s_dict(128, 0);
		vector<int> t_dict(128, 0);
		int l = 0, r = 0;
		int len = s.length();
		int min_len = len + 1;
		int ans_l, ans_r = 0;
		for (int i = 0; i < t.length(); i++) {
			t_dict[t[i]]++;
		}
		while (l < len) {
			while (!contain(s_dict, t_dict) && r < len) {
				s_dict[s[r]]++;
				r++;
			}
			// cout << "r" << r << endl;
			while (contain(s_dict, t_dict)) {
				s_dict[s[l]]--;
				l++;
			}
			// cout << "l" << l << endl;
			if (r - l + 1 < min_len) {
				min_len = r - l + 1;
				ans_l = l - 1;
			}
			if (r >= len) break;
		}
		// cout << min_len << endl;
		if (min_len == len + 1) return "";
		return s.substr(ans_l, min_len);
	}

  private:
	bool contain(vector<int> & s_dict, vector<int>& t_dict) {
		for (int i = 0; i < 128; i++) {
			if (t_dict[i]) {
				if (t_dict[i] > s_dict[i]) return false;
			}
		}
		return true;
	}
};
```

## 方法2

### 思路

滑动窗口的框架一致。改进的地方在于窗口是否符合条件的判断上。

判断的思路是记录总共需要去寻找的字符的个数。比如"AABC"，总共需要4个字符被包含。

使用一个128大小的数组作为哈希表（初始为0），先遍历t字符串得到每个字符的出现次数。然后窗口右端扩展的时候，对应的字符数目减1，当遇到t字符串中的字符的时候，将需要寻找的字符数减1。当需要寻找字符的数目为0的时候，则当前窗口已经包含了t字符串中的字符，于是修改左边开始减小窗口长度。当从左边的窗口开始遍历的时候则对应遍历到的字符技术加1，如果是t字符串中的字符则需要寻找的字符数加1，停止左边窗口的遍历。

总结下：

1. 右边窗口遍历时，对应计数减1，左边窗口遍历时，对应计数加1。类似于出栈入栈的过程。
2. 判断是否为目标字符的方法，当改字符的计数大于0则是目标字符。
3. 通过判断需要寻找的目标字符数来作为窗口的判断条件。

### 复杂度

窗口合法的条件判断的时间复杂度变为了$O(1)$因此整体时间复杂度为$O(N)$,空间复杂度为$O(128)$

### 代码

```cc
/*
Runtime: 20 ms, faster than 80.47% of C++ online submissions for Minimum Window Substring.
Memory Usage: 7.6 MB, less than 87.18% of C++ online submissions for Minimum Window Substring.
 */
class Solution {
  public:
	string minWindow(string s, string t) {
		vector<int> count(128, 0);
		int l = 0, r = 0;
		int len = s.length();
		int min_len = len + 1;
		int ans_l, ans_r = 0;
		for (int i = 0; i < t.length(); i++) {
			count[t[i]]++;
		}
		int len2find = t.length();
		while (l < len) {
			while (len2find && r < len) {
				if (count[s[r]] > 0) len2find--;
				count[s[r]]--;
				r++;
			}
			// cout << "r" << r << endl;
			while (!len2find) {
				count[s[l]]++;
				if (count[s[l]] > 0 ) len2find++;
				l++;
			}
			// cout << "l" << l << endl;
			if (r - l + 1 < min_len) {
				min_len = r - l + 1;
				ans_l = l - 1;
			}
			if (r >= len) break;
		}
		// cout << min_len << endl;
		if (min_len == len + 1) return "";
		return s.substr(ans_l, min_len);
	}
};
```

## 参考

https://leetcode.com/submissions/detail/369501369/
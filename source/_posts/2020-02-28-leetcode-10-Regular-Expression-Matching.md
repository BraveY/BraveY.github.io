---
title: leetcode 10 Regular Expression Matching
date: 2020-02-28 16:07:23
categories: 题解
tags:
- 字符串
- 动态规划
- 递归
copyright: true
---

# leetcode 10 Regular Expression Matching

[题目来源](<https://leetcode.com/problems/regular-expression-matching/> )，需要实现字符串正则匹配的'*'和'.'的匹配功能，判断给定字符串和模式能否匹配。

Given an input string (`s`) and a pattern (`p`), implement regular expression matching with support for `'.'` and `'*'` where:` `

- `'.'` Matches any single character.
- `'*'` Matches zero or more of the preceding element.

The matching should cover the **entire** input string (not partial).

 

**Example 1:**

```
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
```

**Example 2:**

```
Input: s = "aa", p = "a*"
Output: true
Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
```

**Example 3:**

```
Input: s = "ab", p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".
```

**Example 4:**

```
Input: s = "aab", p = "c*a*b"
Output: true
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore, it matches "aab".
```

**Example 5:**

```
Input: s = "mississippi", p = "mis*is*p*."
Output: false
```

 

**Constraints:**

- `0 <= s.length <= 20`
- `0 <= p.length <= 30`
- `s` contains only lowercase English letters.
- `p` contains only lowercase English letters, `'.'`, and `'*'`.
- It is guaranteed for each appearance of the character `'*'`, there will be a previous valid character to match.

## 思路

### 递归

因为字符串去掉一个字符后仍然是字符串，所以可以考虑递归的形式。首先判断字符串s和模式字符串t的第一个字符s[0]和t[0]，如果二者第一个字符相等则接下来考虑第二个字符。

t的第二个字符有两种情况：

s[1...n]表示s字符串的第二个字符一直到结尾的子字符串，t[1...m]表示从第二个字符开始一直到结尾的子字符串。

1. t[1]==* 如果是星号匹配符而且前一个字符t[0]和s[0]匹配 则*可能复制前一个字符t[0]
   1. 0次。即不复制前一个字符 这时模式字符串t相当于去掉前两个字符用剩下的t[2...m]和s字符串去掉第一个匹配的字符剩下的s[1...n]进行匹配
   2. 大于0次。即复制前一个字符至少1次，因为不确定*号具体复制了多少次，这时候应该理解成模式字符串的长度不变，但是消耗了s字符串的一个已经匹配的第一个字符，即结果由t[0...m]和s[1...n]的匹配结果决定
2. 如果t字符串的第二个字符不是*号，则将s和t都去掉第一个字符后用剩下的s[1...n]和t[1...m]匹配。

这种分支情况比较多的递归还是第一次遇到。

### 动态规划

之前的动态规划一般都是从后往前来考虑子问题的，即数组A[0...n-1]去掉最末尾的元素然后变成子问题A[0...n-2]。我最开始也是用这种思维来定义子问题的，用opt\[m][n]表示s字符串长度前m个和t模式字符串前n个字符的结果。然后从s和t只有一个字符的时候开始填充二维表，但是这样的话会有逻辑漏洞，比如s是“aa”和t是"c\*a*"的时候第一个字符就不匹配，会导致前面就没有值为真的时候，无论状态怎么转移，也不可能转移到后面有值为真的状态。

结合上面的递归思路，子问题的定义刚好是**从后面往前**的。最简单的子问题是最末尾只有一个字符的时候，然后才是向前逐个添加字符的。所以定义子问题opt\[m][n] 为s字符串从第m个字符开始，t字符串从第n个字符开始的匹配结果。

然后考虑状态转移方程，依然是第一个字符匹配的情况下：

如果t[j]不是*号：opt\[i][j] 由s和t各去掉第一个字符后的opt\[i+1][j+1]状态转移

如果t[j]是*号但选择不复制：opt\[i][j] 由s去掉第一个字符而t去掉两个字符的状态opt\[i+1][j+2] 转移

如果t[j]是*号并且选择复制：opt\[i][j] 由s去掉第一个字符而t不变的状态opt\[i+1][j] 转移

### 复杂度

两层循环所以时间复杂度是$O(mn)$,二维的空间所以空间复杂度是$O(mn)$

m是s的长度，n是t的长度

## 代码

[代码源文件](<https://github.com/BraveY/Coding/blob/master/leetcode/10regular-expression-matching.cc> )

### 递归

```cc
//有分支的递归
/*
Runtime: 224 ms, faster than 10.31% of C++ online submissions for Regular
Expression Matching. Memory Usage: 15.5 MB, less than 8.48% of C++ online
submissions for Regular Expression Matching.
 */
class Solution {
 public:
  bool isMatch(string s, string p) {
    if (p.empty()) return s.empty();
    bool first_match = (!s.empty() && (s[0] == p[0] || p[0] == '.'));
    if (p.length() >= 2 &&
        p[1] == '*') {  //*从头往后递归的，所以*只有在第二个位置才有意义
      return (isMatch(s, p.substr(2)) ||
              (first_match && isMatch(s.substr(1), p)));
      //*匹配0个，所以向后移动2个字符||*匹配的个数为正数，且第一个字符匹配，所以t-1，而pattern的格式仍然不变，因为*没有取0所以还存在
    } else {  //没有*的情况看后面的是否匹配
      return (first_match && isMatch(s.substr(1), p.substr(1)));
    }
    return false;
  }

 private:
};
```

### 动态规划

```cc
/*
Runtime: 4 ms, faster than 93.47% of C++ online submissions for Regular
Expression Matching. Memory Usage: 8.9 MB, less than 69.49% of C++ online
submissions for Regular Expression Matching.
 */
class Solution2 {
 public:
  bool isMatch(string s, string p) {
    if (p.empty() || s.empty()) return false;
    // int m = s.length(), n = p.length();  为空仍然有匹配的 "" 和 "*"
    std::vector<vector<bool> > opt(m + 1, vector<bool>(n + 1, false));
    opt[m][n] = true;  //代表两个空字符串是否匹配
    for (int i = m; i >= 0; i--) {
      for (int j = n; j >= 0; j--) {
        if (i == m && j == n) continue;  // opt[m][n] 已经初始了
        bool first_match = (i < m && j < n && (s[i] == p[j] || p[j] == '.'));
        if (j + 1 < n && p[j + 1] == '*') {
          opt[i][j] = opt[i][j + 2] || (first_match && opt[i + 1][j]);
        } else {
          opt[i][j] = first_match && opt[i + 1][j + 1];
        }
      }
    }
    return opt[0][0];
  }

 private:
};
```

## 参考

[windliang ](<https://leetcode.wang/leetCode-10-Regular-Expression-Matching.html> )
---
title: leetcode 139 Word Break
date: 2020-02-11 13:09:23
categories: 题解
tags:
- 动态规划
- 递归
copyright: true
---

# leetcode 139 Word Break

[题目来源](<https://leetcode.com/problems/word-break/> ),求解一个字符串能否由给定的子字符串组成。

<!--more-->

## 思路

这道题涉及到字符串的操作，之前不是很熟悉。

## 记忆化递归

输入的是字符串，也是属于可分的数据结构，因此可以考虑分成子问题后使用递归的方法来求解。将字符串分为left和right部分，对于left部分递归调用子问题去求解left这部分是否为真，对于right部分判断是否在字典中。只有left部分的解为真而且right部分在字典中，完整的字符串才是可以被字典中的字符串组成的。因为划分为两部分的方法总共有字符串长度n种，所以需要对n种划分方法都遍历，只要有一种划分方法为真，那么原问题的字符串方法为真。

以“leetcode”为例有：
$$
\begin{equation}\nonumber
\begin{split}
wB("leetcode")&=wB("")\&\&inDict("leetcode")\\
&||wB("l")\&\&inDict("eetcode")\\
&||wB("le")\&\&inDict("etcode")\\
&||wB("lee")\&\&inDict("tcode")\\
&||\textbf{wB("leet")&&inDict("code")}\\
&||wB("leetc")\&\&inDict("ode")\\
&||wB("leetco")\&\&inDict("de")\\
&||wB("leetcod")\&\&inDict("e")\\
\end{split}
\end{equation}
$$
这种递归实质上是一种枚举，如果不优化的话很多子问题会被重复计算，所以需要采用记忆化递归的方法。使用一个memo备忘表，存储每个子问题的解，递归时如果改子问题已经有解，就可以不必重复计算。

### 复杂度

直接分析不好分析转化为动态规划好分析一些见下面,时间复杂度$O(n^2)$,空间复杂度$O(n)$

## 动态规划

### 基础版

动态规划本身就是记忆化递归的一种实现方式，因此既然写出了记忆化递归的递归式，写出动态规划的形式也就比较容易了。将最优子结构定义为opt[i]:表示字符串前i个字符构成的字符串能否由字典中的字符串组成。对比着上面的递归表达式就可以写出状态转移方程：
$$
if\;\{opt[j]\&\&inDict(s.substing(j,i-j))\quad j<i\}\\
opt[i] = true\;
$$
也就是opt[i]的状态需要去查看其前面的每一个状态opt[j],如果前面有一个状态为真，而且剩下的right部分也为真那么此时的状态为真。

### 复杂度

这个版本的时间复杂度，两层循环因此为$O(n^2)$,空间复杂度$O(n)$

### 优化版

基础版的状态转移方式是一个字符一个字符的去遍历的，但是实际上只有按照字典里面字符串的长度去探索才有可能得到真的情况。比如样例s="leetcode"，wordDict={"leet","code"}，字典中字符串的长度为4，因此只有对长度为4的字符串去判断inDict才有意义。比如对left=”leetco“和right=”de“去判断是否为真是没有意义的，因为”de“肯定不在字典中的。

所以新的状态转移方程为：
$$
if\;\{opt[i-len]\&\&inDict(word))\quad len<i\}\\
opt[i] = true\;
$$
word是长度为len的字符串。

如果按照背包问题来理解的话，就是考虑每次选择长度为len的字符串right，然后判断inDict(right)&&opt[i-len]。len的取值为wordDict中的所有字符串的长度。

### 复杂度

时间复杂度为O(mn),m为字典的长度，空间复杂度O(n)。

## 代码

[代码源文件](https://github.com/BraveY/Coding/blob/master/leetcode/139word-break.cpp)

### 记忆化递归

参考花花的代码，值得学习的地方：

1. memo使用hash表来存储，然后通过哈希表的函数dict(key)，来判断key是否存在。
2. 将wordDict也重新用哈希表来存储，方便快速确定一个字符串是否在字典中
3. string.substr(loc,len):从loc开始截取len长度的字符串，只有loc，则是从loc一直截取到尾部。

```cc
/*
Runtime: 16 ms, faster than 50.07% of C++ online submissions for Word Break.
Memory Usage: 15.9 MB, less than 24.53% of C++ online submissions for Word
Break.
 */
class Solution1 {
 public:
  bool wordBreak(string s, vector<string>& wordDict) {
    //将wordDict中的的元素记录在hashset中，方便查找一个字符串是否在wordDict中
    unordered_set<string> dict(wordDict.cbegin(), wordDict.cend());
    return wordBreak(s, dict);  //重载
  }

  bool wordBreak(const string& s, const unordered_set<string>& dict) {
    if (memo.count(s)) return memo[s];  //已经求过s的答案 直接返回
    if (dict.count(s)) return memo[s] = true;  // s就在wordDict中
    for (int j = 1; j < s.length(); j++) {
      const string left = s.substr(0, j);
      const string right = s.substr(j);
      if (dict.count(right) && wordBreak(left, dict)) return memo[s] = true;
    }
    return memo[s] = false;
  }

 private:
  unordered_map<string, bool> memo;
};
```

### 基础DP

```cc
/*
Runtime: 12 ms, faster than 61.68% of C++ online submissions for Word Break.
Memory Usage: 15.7 MB, less than 26.41% of C++ online submissions for Word
Break.
 */
class Solution2 {
 public:
  bool wordBreak(string s, vector<string>& wordDict) {
    //将wordDict中的的元素记录在hashset中，方便查找一个字符串是否在wordDict中
    unordered_set<string> dict(wordDict.cbegin(), wordDict.cend());
    int n = s.length();
    std::vector<bool> opt(n + 1);
    opt[0] = true;
    for (int i = 1; i <= n; i++) {
      // const string sub = s.substr(0, i);
      for (int j = 0; j < i; j++) {
        // const string right = sub.substr(j);
        if (dict.count(s.substr(j, i - j)) && opt[j]) {
          opt[i] = true;
          break;
        }
      }
    }
    return opt[n];
  }
};
```

### 优化DP

```cc
/*
Runtime: 0 ms, faster than 100.00% of C++ online submissions for Word Break.
Memory Usage: 8.9 MB, less than 92.45% of C++ online submissions for Word Break.
 */
class Solution3 {
 public:
  bool wordBreak(string s, vector<string>& wordDict) {
    //将wordDict中的的元素记录在hashset中，方便查找一个字符串是否在wordDict中
    unordered_set<string> dict(wordDict.cbegin(), wordDict.cend());
    int n = s.length();
    int m = wordDict.size();
    std::vector<bool> opt(n + 1);
    opt[0] = true;
    for (int i = 1; i <= n; i++) {
      // const string sub = s.substr(0, i);
      for (int j = 0; j < m; j++) {
        int len = wordDict[j].length();
        if (len <= i && opt[i - len] &&
            dict.count(s.substr(i - 1 - len + 1, len))) {
          opt[i] = true;
          break;
        }
      }
    }
    return opt[n];
  }
};
```

## 参考

[花花leetcode](<https://v.youku.com/v_show/id_XMzgwMjQ2MzEyNA==.html> )

[lucifer](<https://github.com/azl397985856/leetcode/blob/master/problems/139.word-break.md> )
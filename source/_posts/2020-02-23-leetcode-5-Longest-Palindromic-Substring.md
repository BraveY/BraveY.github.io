---
title: leetcode 5 Longest Palindromic Substring
date: 2020-02-23 16:52:07
categories: 题解
tags:
- 动态规划
- 马拉车算法
copyright: true
---

# leetcode 5 Longest Palindromic Substring

[题目来源](<https://leetcode.com/problems/longest-palindromic-substring/> ),要求出一个字符串中的最长回文子字符串。

## 思路

### 动态规划

自己最开始考虑的子问题定义是opt[i]表示以s[i]结尾的最长回文字符串，用的是一个一维的动态规划数组opt[i]来存储状态，这样行不通原因是与前一个字符形成的回文的可能是所有状态的。比如回文是1到i-1则与i形成回文的范围是从0到i，都是有可能的，而不是opt[i]简单的只从前面opt[i-1]转移。

因此需要存储前面每一种长度的字符串是否为回文的状态，将子问题的形式定义为：从i开始到j结束的字符串是否为回文，使用opt\[i][j]二维数组来存储。定义好子问题的形式后考虑如何填充二维动态规划表，也就是状态转移方程。总共有三种情况：

1. 长度为1的字符串本身就是一个回文字符串
2. 长度为2的字符串如果二者相等则也是一个回文字符串
3. opt\[i+1][j-1]为true，并且s[i]与s[j]相等则opt\[i][j]为true

$$
opt[i][j]=
\begin{cases}
true& \text{j-i=0}\\
true& \text{j-i=1 and s[i]==[sj]}\\
true& \text {opt[i+1][j-1]==true and s[i]==s[j]}\\
\end{cases}
$$

因此使用一个双重循环去构建表格

#### 复杂度

双重循环所以时间复杂度为$O(n^2)$,二维表格因此空间复杂度为$O(n^2)$

### 最长公共子串

因为回文的对称性，将输入字符串逆向得到字符串t，然后问题可以转换为求两个字符串的最长公共子串。与[1143](<https://leetcode.com/problems/longest-common-subsequence/> ) 求最长公共子序列有点类似，依然是态规划，之前做过但没有写题解，后面找时间补上，1143[代码](<https://github.com/BraveY/Coding/blob/master/leetcode/longest-common-subsequence.cc> )参考。

### 中心扩展法

从上面第三个情况得到一个启示，回文串的核心特征是对称，偶数长度的对称中心在两个字符中间，奇数长度的对称中心在中间的字符。从对称中心开始向两边扩展，如果两边字符相等则构成回文字符串。

因此遍历字符串，对每个字符求解以它为中心（奇数长度）或者它和下个字符中间为中心的（偶数长度）的最长回文字符串。遍历一遍后得到最长的回文字符串。

#### 复杂度

外层一次遍历，每个字符又有一次遍历，所以两层循环时间复杂度$O(n^2)$,不需要额外的空间，空间复杂度$O(n)$

### 马拉车算法

该算法将时间复杂度下降到了线性，参考[一文让你彻底明白马拉车算法]( https://zhuanlan.zhihu.com/p/70532099 )，简单记录下自己的理解：

1. 将原字符串填充后扩展为2*n+1的奇数长度
2. 在回文字符串里面的且长度没有超过回文字符串的字符通过对称直接得到
3. 超过了回文字符串范围的使用中心扩展法来求解
4. 对称中心更新条件：新求出的对称长度超过原来的对称中心的回文字符串范围就更新。

#### 复杂度

因为每个使用中心扩展法的遍历的字符，下次可以直接利用对称得到自己的解，所以每个节点使用中心扩展法的机会只有一次，所以是$O(n)$，新建了一个新的字符串，所以空间复杂度为$O(n)$.

## 代码

[代码源文件](https://github.com/BraveY/Coding/blob/master/leetcode/5longest-palindromic-substring.cc)

### 动态规划

```cc
/*
Runtime: 336 ms, faster than 12.50% of C++ online submissions for Longest
Palindromic Substring. Memory Usage: 26.5 MB, less than 20.69% of C++ online
submissions for Longest Palindromic Substring.
 */
class Solution {
 public:
  //只用一维opt[i]来表示不可行，因为与前一个字符形成的回文的可能是所有状态的。比如回文是1到i-1
  //则与i形成回文的范围是从0到i。都是有可能的
  string longestPalindrome(string s) {
    int n = s.length();
    if (n == 0 || n == 1) return s;
    string ans = s.substr(0, 1);
    vector<vector<bool>> opt(n, vector<bool>(n));
    for (int i = n - 1; i >= 0; i--) {
      for (int j = i; j < n; j++) {
        if (j - i == 0)
          opt[i][j] = true;                         //单个字符构成回文
        else if (j - i == 1 && s.at(i) == s.at(j))  //两个相邻 相等字符构成回文
          opt[i][j] = true;
        else if (s.at(i) == s.at(j) && opt[i + 1][j - 1])
          opt[i][j] = true;
        if (opt[i][j] && j - i + 1 > ans.length())
          ans = s.substr(i, j - i + 1);  //长度比现有答案的长度大就更新。
      }
    }
    return ans;
  }

 private:
};
```

### 中心扩展法

```cc
/*
Runtime: 16 ms, faster than 86.12% of C++ online submissions for Longest
Palindromic Substring. Memory Usage: 8.8 MB, less than 78.62% of C++ online
submissions for Longest Palindromic Substring.
 */
class Solution2 {
 public:
  string longestPalindrome(string s) {
    const int n = s.length();
    auto getLen = [&](int l, int r) {  // 计算每个字符可扩展到的长度
      while (l >= 0 && r < n && s[l] == s[r]) {
        --l;
        ++r;
      }
      return r - l - 1;
    };
    int len = 0;
    int start = 0;
    for (int i = 0; i < n; ++i) {
      //(i,i)以自身为起点 （i，i+1）以中间为起点
      int cur = max(getLen(i, i), getLen(i, i + 1));
      if (cur > len) {
        len = cur;
        start = i - (len - 1) / 2;
      }
    }
    return s.substr(start, len);
  }

 private:
};
```

getLen是C++的lamda表达式。

### 马拉车算法

```cc
/*
马拉车
Runtime: 24 ms, faster than 70.00% of C++ online submissions for Longest
Palindromic Substring. Memory Usage: 141.8 MB, less than 8.96% of C++ online
submissions for Longest Palindromic Substring.
 */
class Solution3 {
 public:
  string longestPalindrome(string s) {
    string T = padding(s);
    int n = T.length();
    std::vector<int> P(n);
    int C = 0, R = 0;
    for (int i = 1; i < n - 1; i++) {  //原字符串的开始长度1和n-2
      int i_mirror = 2 * C - i;
      if (R > i) {
        P[i] = min(R - i, P[i_mirror]);  //防止超出R
      } else {
        P[i] = 0;
      }
      //中心扩展法都要使用
      while (T.at(i + 1 + P[i]) == T.at(i - 1 - P[i])) {
        P[i]++;
      }
      if (i + P[i] > R) {
        C = i;
        R = i + P[i];
      }
    }
    int max_len = 0;
    int center = 0;
    for (int i = 1; i < n - 1; i++) {
      if (P[i] > max_len) {
        max_len = P[i];
        center = i;
      }
    }
    int start = (center - max_len) / 2;  //原始字符串下标
    return s.substr(start, max_len);
  }

 private:
  string padding(string& s) {
    int n = s.length();
    if (n == 0) return "^$";
    string rtn = "^";
    for (int i = 0; i < n; i++) {
      rtn = rtn+"#" + s.at(i); //不能使用+= 
    }
    rtn += "#$";
    return rtn;
  }
};
```

马拉车并不比原来的中心扩展法快，可能是样例影响使得填充的时间要多一些。

## 参考

[花花leetcode](<https://zxi.mytechroad.com/blog/dynamic-programming/leetcode-5-longest-palindromic-substring/> )

[lucifer](<https://github.com/azl397985856/leetcode/blob/master/problems/5.longest-palindromic-substring.md>  )
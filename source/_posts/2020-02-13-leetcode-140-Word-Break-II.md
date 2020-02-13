---
title: leetcode 140 Word Break II
date: 2020-02-13 11:13:45
categories: 题解
tags:
- 记忆化递归
- 动态规划
copyright: true
---

# leetcode 140 Word Break II

[题目来源](<https://leetcode.com/problems/word-break-ii/> ),要求给出字符串由字典中的子字符串的所有组成方式。

## 思路

同[Word Break](<https://bravey.github.io/2020-02-11-leetcode-139-Word-Break.html#more> )的思路大致一致，也是有两种方法，但是动态规划有个样例被卡了无法通过。

### 记忆化递归

整体思路依然是将输入字符串s分成左边与右边两部分，先对右边的判断是否在字典中，如果是的话在递归的去寻找做左边的答案，然后将左边的答案各种组成方式加上在字典中的右边部分，作为s的答案。

### 动态规划(TLE)

动态规划的思路同[Word Break](<https://bravey.github.io/2020-02-11-leetcode-139-Word-Break.html#more> )大体一致，转移方程也是：
$$
if\;\{opt[i-len]\&\&inDict(word))\quad len<i\}\\
opt[i] = true\;
$$
只是额外开了个数组来存储不同长度的字符串的答案。然后在每次能够成功匹配的时候在上个状态的答案下去追加右边right部分。

动态规划从更断的状态一步一步来填充答案，会造成一个问题：当样例为：

```cc
  const string s =
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "aaaaaaa";
  vector<string> wordDict({"a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa",
                           "aaaaaaa", "aaaaaaaa", "aaaaaaaaa", "aaaaaaaaaa"});
```

的时候，因为从短到长填充答案的时候，还没到判断后面的b的时候就会超时了。因为'b'前面的字符串都是有解得，所以一步一步填充的时候会非常慢。而记忆化递归有个首先先判断是否在dict中的逻辑，对于这样没有解的情况判断非常快。如果是"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"这样的样例记忆化递归也依然会TLE。

## 代码

[代码源文件](<https://github.com/BraveY/Coding/blob/master/leetcode/140word-break-ii.cc> )

### 记忆化递归

```cc
/*
https://zxi.mytechroad.com/blog/leetcode/leetcode-140-word-break-ii/
Runtime: 16 ms, faster than 69.32% of C++ online submissions for Word Break II.
Memory Usage: 15.1 MB, less than 54.55% of C++ online submissions for Word Break
II.
 */
class Solution {
 public:
  vector<string> wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> dict(wordDict.cbegin(), wordDict.cend());
    return wordBreak(s, dict);
  }

 private:
  // >> append({"cats and", "cat sand"}, "dog");
  // {"cats and dog", "cat sand dog"}
  vector<string> append(const vector<string>& left, const string& right) {
    vector<string> rtn;
    //左边为空的时候，不会加上右边，返回的也是空
    for (const auto& iter : left) rtn.push_back(iter + " " + right);
    return rtn;
  }

  const vector<string> wordBreak(string s, unordered_set<string>& dict) {
    if (memo.count(s)) return memo[s];
    int n = s.length();
    // 空字符串的处理？空字符串默认为vector也为空，所以不用处理

    //记录答案
    vector<string> ans;
    //如果完整字符串在字典里则直接加入到答案，
    //之所以提出来写是因为wordBreak("")为空，因此直接加上right，
    //不加上“ ”,不使用append函数
    if (dict.count(s)) ans.push_back(s);
    //不从0开始分割，为0的情况在上面if语句中已经判断过了
    for (int i = 1; i < n; i++) {
      const string& right = s.substr(i);
      //先判断右边是否在字典中，这也是记忆化递归能比动态规划快的原因，
      //不会先去求解，从而造成TLE，而是先判断是否需要求解
      if (dict.count(right)) {
        const string& left = s.substr(0, i);
        const vector<string> left_ans =
            append(wordBreak(left, dict), right);  //左边的结果加上新的末尾
        //不能使用
        // memo[s]来填充,因为还没算完,只是其中一种解。所以后续的递归如果访问了memo[s]，结果是不一致的
        // memo[s].swap(
        //     append(left_ans, right));
        ans.insert(ans.end(), left_ans.begin(),
                   left_ans.end());  //将left_ans的答案添加到ans
      }
    }
    memo[s].swap(ans);
    return memo[s];
  }

 private:
  unordered_map<string, vector<string>> memo;
};
```

### 动态规划TLE

```cc
//动态规划从小到大求解状态会造成花过多时间和内存在计算前面状态，如果为0的状态在很后面则不能快速求解，从而TLE
class TLESolution {
 public:
  vector<string> wordBreak(string s, vector<string>& wordDict) {
    int n = s.length();
    int m = wordDict.size();
    // std::vector<vector<string>> memo(n + 1);
    unordered_map<int, vector<string>> memo;
    unordered_set<string> dict(wordDict.cbegin(), wordDict.cend());
    set<int> lens;
    for (int i = 0; i < m; i++) {
      lens.insert(wordDict[i].length());
    }
    std::vector<bool> opt(n + 1);
    opt[0] = true;
    for (int i = 1; i <= n; i++) {
      for (auto j = lens.begin(); j != lens.end(); j++) {
        // int len = wordDict[j].length();
        int len = *j;
        if (len <= i) {
          const string word = s.substr(i - 1 - len + 1, len);
          if (opt[i - len] && dict.count(word)) {
            opt[i] = true;
            int l = memo[i - len].size();
            if (!l)
              memo[i].push_back(word);
            else {
              for (int k = 0; k < l; k++) {
                memo[i].push_back(memo[i - len][k] + " " + word);
              }
            }
          }
        }
      }
    }
    return memo[n];
  }

 private:
};
```


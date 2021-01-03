---
title: leetcode 438 Find All Anagrams in a String
date: 2021-01-03 21:29:17
categories: 题解
tags:
- 滑动窗口
- 哈希表
copyright: true
---

## 题意

求一个字符串的所有字母组合在另一个字符串中的开始下标。

Given a string **s** and a **non-empty** string **p**, find all the start indices of **p**'s anagrams in **s**.

Strings consists of lowercase English letters only and the length of both strings **s** and **p** will not be larger than 20,100.

The order of output does not matter.

**Example 1:**

```
Input:
s: "cbaebabacd" p: "abc"

Output:
[0, 6]

Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
```



**Example 2:**

```
Input:
s: "abab" p: "ab"

Output:
[0, 1, 2]

Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
```

[题目链接]()

## 方法1 暴力+哈希表

### 思路

判断一个字符串是否为另一个字符串的组合，比较容易想到将字母存入哈希表中，然后比较每个字母的个数是否一致。之后s中对每个窗口长度的子字符串都进行遍历建立字母哈希表来进行比较。

### 复杂度

时间复杂度$O(N^2)$

空间复杂度$O(26*2)$

### 代码

```cc
/*
Runtime: 852 ms, faster than 5.22% of C++ online submissions for Find All Anagrams in a String.
Memory Usage: 31.1 MB, less than 5.26% of C++ online submissions for Find All Anagrams in a String.
*/

class Solution1 {
  public:    
    vector<int> findAnagrams(string s, string p) {
        int m = s.size();
        int n = p.size();
        if(!m || !n || m < n) return {};
        for(int i = 0; i < n; ++i){
            dict[p[i] - 'a']++;
        }
        vector<int> ans;
        for(int i = 0; i <= m - n; ++i){
            if(isAnagrams(s, i, i + n - 1))
                ans.push_back(i);
        }
        return ans;
    }

  private:    
    vector<int> dict = vector<int>(26, 0);
    bool isAnagrams(string&s, int iStart, int iEnd){
        vector<int> dictLocal(26, 0);
        for(int i = iStart; i <= iEnd; ++i){
            dictLocal[s[i] - 'a']++;
        }
        for(int i = 0; i < 26; ++i){
            if(dict[i] != dictLocal[i])
                return false;
        }
        return true;
    }
};
```

## 方法2 滑动窗口+哈希表

### 思路

方法1每次都重新建立哈希表，实际上每次窗口移动的时候除了左右两边的字母会发生变化，中间剩下的字母个数不会变化，也就不需要再重新建立。这也就是滑动窗口的思想，当长度没有达到窗口大小的时候，右边依次递增并将字母放入哈希表中，当超过窗口长度的时候，最左边的字母从哈希表中递减。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(26*2)$

### 代码

```cc
/*
Runtime: 20 ms, faster than 56.38% of C++ online submissions for Find All Anagrams in a String.
Memory Usage: 9 MB, less than 38.52% of C++ online submissions for Find All Anagrams in a String.
滑动窗口的字典是动态更新的，只需要加入新进入窗口的，减少不是窗口的状态就可以了。暴力的每次都需要重新建立。
*/
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int m = s.size();
        int n = p.size();
        vector<int> ans;
        vector<int> vs(26, 0);
        vector<int> vp(26, 0);
        for(int i = 0; i < n; ++i) {
            ++vp[p[i] - 'a'];
        }
        for(int i = 0; i < m; ++i) {
            if ( i >= n) {
                --vs[s[i - n] - 'a'];
            }
            ++vs[s[i] - 'a'];
            if( vs == vp) ans.push_back( i - n + 1);
        }
        return ans;
    }
};
```

## 总结

1. 滑动窗口的精髓在于动态的去更新滑动窗口中的状态，不足窗口长度的时候只有右边的元素进入窗口，超过窗口长度的时候每次移动都会有左边的元素状态递减，右边的元素状态递增。

## 参考

[huahua](https://www.bilibili.com/video/av31292369)
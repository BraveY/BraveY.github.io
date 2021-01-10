---
title: leetcode 32 Longest Valid Parentheses
date: 2021-01-10 16:16:20
categories: 题解
tags:
- 栈
- 动态规划
copyright: true
---

## 题意

求能够满足括号匹配有效性的最长子字符串。

[题目链接](https://leetcode.com/problems/longest-valid-parentheses/)

Given a string containing just the characters `'('` and `')'`, find the length of the longest valid (well-formed) parentheses substring.

**Example 1:**

```
Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()".
```

**Example 2:**

```
Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".
```

**Example 3:**

```
Input: s = ""
Output: 0
```

**Constraints:**

- `0 <= s.length <= 3 * 104`
- `s[i]` is `'('`, or `')'`.

## 方法1

### 思路

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

## 方法1 暴力+栈

### 思路

遍历所有子字符串对，然后对每个字符串用栈来判断是否满足括号匹配的有效性，如果满足则去比较更新最大值。

### 复杂度

时间复杂度$O(N^3)$

空间复杂度$O(N)$

### 代码

```cc
/*
O(n^3)
Time Limit Exceeded
*/
class Solution1 {
  public:
    int longestValidParentheses(string s) {
        int ans = 0;
        for(int i = 0; i < s.size(); ++i){
            for(int j = 0; j <= i; ++j){
                if(isvalid(j, i, s)){
                    ans = max(ans, i - j + 1);
                }
            }
        }
        return ans;
    }

  private:
    bool isvalid(int l, int r, string str ){
        if (l >= r) return false;
        stack<int> s;
        for(int i = l; i <= r; ++i ) {
            if(str[i] == '(') s.push(i);
            else if(str[i] == ')') {
                if(s.empty()) return false;
                else s.pop();
            }
        }
        return s.empty() ? true : false;
    }
};
```

## 方法2 栈

### 思路

类似于移动窗口的方法，只需扫描一遍并动态的更新栈。核心点在于因为栈中存储的是索引值，所以每次有右括号j进行匹配的时候,消耗的栈中的左括号i，二者之间的字符串s[i...j]是有效的，更进一步的i与栈顶的左括号k之间的子字符串s[k+1...j]也是有效的。

比如

![](https://res.cloudinary.com/bravey/image/upload/v1610270551/blog/coding/lc32.jpg)

当遍历到j=6的时候，栈中的元素为`0,3`，匹配上i=3的左括号，栈顶还剩下k=0的左括号。这个时候从1到6的子字符串是有效的。

没有想到这一点也是自己没有做出来的原因。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 8 ms, faster than 39.53% of C++ online submissions for Longest Valid Parentheses.
Memory Usage: 7.7 MB, less than 64.32% of C++ online submissions for Longest Valid Parentheses.
*/
class Solution2 {
public:
    int longestValidParentheses(string s) {
        stack<int> iStack;
        int l = 0;
        int ans = 0;
        int n = s.size();
        for(int i = 0; i < n; ++i){
            if(s[i] == '(') {
                 iStack.push(i);
            }else{
                if(iStack.empty()) l = i + 1;//栈中为空的时候出现了右括号无法匹配，所以移动到下一个节点
                else{
                iStack.pop();
                    //如果匹配后为空则从开始处（包含）计数，否则从栈顶处（不包含）计数
                ans = max(ans, iStack.empty() ? i - l + 1 : i - iStack.top());
                }
            }
        }        
        return ans;
    }
};
```

## 方法3 动态规划

### 思路

定义`dp[i]`为以i结尾的最长有效子字符串，剩下的就是分情况进行状态的讨论了。

如果`s[i]=='('` 则当前dp值为0，不存在以左括号结尾的有效子字符串。 `dp[i] = 0`

当`s[i]==')'`的时候分为两种情况可以构成有效括号匹配。

1. 如果`s[i - 1] == '('` 则当前左括号与前一个的右括号匹配，此时当前值由更前面的`dp[i - 2]` 状态加上2转移而来。即可以与`s[i - 2]`结尾的有效字符串直接相连。
2. 如果`s[i - 1] == ')'` 则当前左括号可以与除去`dp[i-1]`长度的`s[i - dp [i - 1] - 1]`上的右括号匹配，并且同样的可以与更前一位结尾的有效字符串再次相连匹配。

上面图的状况就是第二种例子的匹配情况：j与i匹配构成第一段，第二段则是更左边的1和2的有效匹配字符串。

### 复杂度

时间复杂度$O(N)$

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 0 ms, faster than 100.00% of C++ online submissions for Longest Valid Parentheses.
Memory Usage: 7.7 MB, less than 64.32% of C++ online submissions for Longest Valid Parentheses.
*/
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.length();
        vector<int> dp(n, 0);
        int ans = 0;
        for(int i = 1; i < n; ++i){
            if(s[i] == '(') continue;
            //后面为s[i] == ')
            if(s[i - 1] == '(') { // 与前一个匹配 （）  
                dp[i] = i - 2 >= 0 ? dp[i - 2] + 2 : 2;
            }else{// 与更早之前的匹配(...)
                int j = i - dp[i - 1] - 1; //去除中间的匹配段之后的前面的左括号(
                if( j >= 0 && s[j] == '(' ) { // 最前面的(之前也有匹配的。
                    dp[i] = j - 1 >= 0 ? dp [i - 1] + 2 + dp[j - 1] : dp[i - 1] + 2;
                }
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
```

## 总结

1. 栈需要理解到匹配的栈中的索引就可以计算有效的长度
2. 动态规划则需要把所有转移的状态找出来

## 参考

[栈](https://zxi.mytechroad.com/blog/stack/leetcode-32-longest-valid-parentheses/)

[dp](https://leetcode.com/problems/longest-valid-parentheses/discuss/14133/My-DP-O(n)-solution-without-using-stack)
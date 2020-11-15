---
title: leetcode 406 Queue Reconstruction by Height
date: 2020-11-09 20:28:16
categories: 题解
tags:
- 排序
- 贪心
copyright: true
---

## 题意

[题目链接]()[https://leetcode.com/problems/queue-reconstruction-by-height/]

Suppose you have a random list of people standing in a queue. Each person is described by a pair of integers `(h, k)`, where `h` is the height of the person and `k` is the number of people in front of this person who have a height greater than or equal to `h`. Write an algorithm to reconstruct the queue.

**Note:**
The number of people is less than 1,100.

 

**Example**

```
Input:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

Output:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```

## 方法1 

### 思路

首先对队列按照高度降序排列，同一高度下按照人数降序排列（[7,0],[7,1]），之后从高到低安排顺序（贪心，优先安排高个的）。安排的方法是以第二个值为索引，在第索引个元素前插入当前元素。

### 复杂度

时间复杂度$O(NlogN)$ 快排的时间复杂度。

空间复杂度$O(N)$

### 代码

```cc
/*
Runtime: 256 ms, faster than 64.85% of C++ online submissions for Queue Reconstruction by Height.
Memory Usage: 13 MB, less than 5.20% of C++ online submissions for Queue Reconstruction by Height.
*/
bool cmp(vector<int>& p1, vector<int>& p2){
    return p1[0] > p2[0] || (p1[0] == p2[0] && p1[1] < p2[1]); // true 的顺序与最终排列结果一致
}
class Solution {
  public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        //sort(people.begin(), people.end(),cmp); 传统方式
        sort(people.begin(), people.end(),[](vector<int>& p1, vector<int>& p2){ // lambda 表达式 https://www.cnblogs.com/DswCnblog/p/5629165.html
            return p1[0] > p2[0] || (p1[0] == p2[0] && p1[1] < p2[1]);
        });
        vector<vector<int>> rtn;
        for(auto person : people){
            rtn.insert(rtn.begin() + person[1], person); // 在index 前插入 person元素
        }
        return rtn;
    }

  private:
};
```

可以使用lambda表达式来写cmp比较函数。

## 总结

1. 要排序是很容易想到的，主要是根据哪一个来排序不好想到。
2. 后面安排的方法是根据人数来的，因为人数本身带有一部分索引的信息。

## 参考

[disscus](https://leetcode.com/problems/queue-reconstruction-by-height/discuss/89345/Easy-concept-with-PythonC%2B%2BJava-Solution)
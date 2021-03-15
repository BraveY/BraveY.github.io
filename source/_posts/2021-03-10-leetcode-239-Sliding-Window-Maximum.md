---
title: leetcode 239 Sliding Window Maximum
date: 2021-03-10 14:31:11
categories: 题解
tags:
- 队列
- BST
copyright: true
---

## 题意

[题目链接](https://leetcode.com/problems/sliding-window-maximum/)

You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.

Return *the max sliding window*.

**Example 1:**

```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**Example 2:**

```
Input: nums = [1], k = 1
Output: [1]
```

**Example 3:**

```
Input: nums = [1,-1], k = 1
Output: [1,-1]
```

**Example 4:**

```
Input: nums = [9,11], k = 2
Output: [11]
```

**Example 5:**

```
Input: nums = [4,-2], k = 2
Output: [4]
```

 

**Constraints:**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`
- `1 <= k <= nums.length`

## 方法1 暴力

### 思路

没次都对窗口从头到尾进行遍历从而得到当前窗口的最大值

### 复杂度

时间复杂度$O((N-k+1)K)$

空间复杂度$O(1)$

### 代码

```cc
/*
O(NK)
Time Limit Exceeded
*/
class Solution1 {
  public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ret;
        int l = 0;
        int r = l + k - 1;
        while(r < nums.size()){
            ret.push_back(*max_element(nums.begin() + l, nums.begin() + r + 1)); // [first, last)
            l++;
            r++;
        }
        return ret;
    }

  private:
};
```

另外可以保存上一个最大值每次与左右两边进行比较，从而使部分情况的获取窗口最大值从$O(K)$变为$O(1)$,但是最坏情况仍然需要遍历窗口（最后一个递减样例就是这种情况）

```cc
/*
最坏情况仍然是O(NK)
Time Limit Exceeded 
*/
class Solution2 {
  public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ret;
        int l = 0;
        int r = l + k - 1;
        int maxLast = *max_element(nums.begin() + l, nums.begin() + r + 1);
        ret.push_back(maxLast);
        l++;
        r++;
        while(r < nums.size()){
            if ( nums[r] >= maxLast ){
                maxLast = nums[r];
                ret.push_back(maxLast);
            } else {
                if ( nums[l - 1] < maxLast) {
                    ret.push_back(maxLast);
                }else {
                    maxLast = *max_element(nums.begin() + l, nums.begin() + r + 1);
                    ret.push_back(maxLast);
                }
            }
            l++;
            r++;
        }
        return ret;
    }

  private:
};
```

## 方法2

### 思路

上述对窗口进行$O(K)$的查询超时，更进一步只有时间复杂度为$O(log(K))$和$O(1)$的解法才可以了，从$O(log(K))$的时间复杂度可以联想到二叉搜索树，因此可以将窗口的值使用BST进行存储。

C++中的map, multimap, set,multimap等容器就是使用红黑树实现的。而红黑树是一颗平衡的二叉搜索树，能够保证查询，插入，删除的平均时间复杂度为$O(log(K))$。有因为这些关联容器存储的元素都是有序的，所以可很方便的返回最大、最小值。

所以可以采用multiset来存储队列的值，并同步的进行插入，删除，查询最大值的操作。

### 复杂度

时间复杂度$O((N – K + 1) * log(K))$

空间复杂度$O(K)$

### 代码

```cc
/*
set的
Time complexity: O((n – k + 1) * logk)
Space complexity: O(k)
Runtime: 572 ms, faster than 19.86% of C++ online submissions for Sliding Window Maximum.
Memory Usage: 185.7 MB, less than 6.48% of C++ online submissions for Sliding Window Maximum.
*/
class Solution3 {
  public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ret;
        if (nums.empty()) return {};
        multiset<int> window(nums.begin(), nums.begin() + k - 1);  //容器范围初始[first, last) 总共有k-1个元素初始进去
        for (int i = k - 1; i < nums.size(); ++i) {
            window.insert(nums[i]);
            ret.push_back(*window.rbegin()); // multiset是有序的，所以逆序的begin 就是最大值
            if ( i - k + 1 >= 0){ // i 是右边的索引 减去窗口长度加上1是左边的索引
                window.erase(window.equal_range(nums[i - k + 1]).first); //erase() 删除所有值相同的元素，但只能删除一个
            }
        }
        return ret;
    }

  private:
};

```

## 方法3

### 思路

滑动窗口可以理解成为队列，因此问题转化为在$O(1)$的时间复杂度下获得队列的最大值。

使用单调队列来保存队列中可能成为最大值的元素，而不将所有滑动窗口的值保存进队列。其push的原则为：

1. push的元素小于等于队尾的元素，则push进入队尾。因为当队顶的最大元素pop出去后，后面的元素就可以成为最大值了
2. 反之，如果push的元素比队尾大的时候，则队尾元素pop。

也就是将一个元素push进入队列的时候，所有比这个元素小的队列元素都需要pop出去。

队列的pop原则为当滑动窗口左端的元素移除并且刚好是单调队列中的队顶最大元素时，单调队列执行pop操作，最大元素出队。



因为单调队列的首尾两端都需要执行pop操作，所以使用双端队列容器deque来实现。

### 复杂度

时间复杂度$O(N)$, 虽然单调队列push一个大值会需要执行K次的pop_back()，但是执行K次pop_back()的前提也意味着之前有K次的push只需要执行1次push_back()，所以平均下来是O(1)的操作。

空间复杂度$O(K)$

### 代码

```cc
class MonotonicQueue {
  public:
    void push(int val) {
        while(!data.empty() && val > data.back()) { // 只要比队尾的元素大就删除，相等也会进入队列
          data.pop_back();
        }
        data.push_back(val); 
    }
    void pop() {
      data.pop_front();
    }
    int getMax() const { // const 修饰表示只对成员变量进行只读访问
        return data.front(); // 队头是最大的
    }
  private:
    deque<int> data;
};

/*
Runtime: 248 ms, faster than 70.97% of C++ online submissions for Sliding Window Maximum.
Memory Usage: 131.6 MB, less than 23.21% of C++ online submissions for Sliding Window Maximum.
O(N)
*/
class Solution4 {
  public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ret;
        MonotonicQueue q;
        for (int i = 0; i < nums.size(); ++i) { // i是右侧的值
          q.push(nums[i]);
          if (i - k + 1 >= 0) { //左侧大于等于0 表示窗口形成
            ret.push_back(q.getMax());
            if (nums[i - k + 1] == q.getMax()) { //滑动窗口要移除的值等于队列最大值时队列才会pop
              q.pop();
            }
          }
        }
        return ret;
    }

  private:
};
```

可以不单独使用一个类而使代码更精简

```cc
/*
Runtime: 236 ms, faster than 73.71% of C++ online submissions for Sliding Window Maximum.
Memory Usage: 131.9 MB, less than 21.58% of C++ online submissions for Sliding Window Maximum.
*/
class Solution {
  public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ret;
        deque<int> q;
        for (int i = 0; i < nums.size(); ++i) { // i是右侧的值          
          while (!q.empty() && nums[i] > q.back()) {
            q.pop_back();
          }
          q.push_back(nums[i]);
          if (i - k + 1 >= 0) { //左侧大于等于0 表示窗口形成
            ret.push_back(q.front());
            if (nums[i - k + 1] == q.front()) { //滑动窗口要移除的值等于队列最大值时队列才会pop
              q.pop_front();
            }
          }
        }
        return ret;
    }

  private:
};
```

## 总结

1. 主要是对其他数据结构BST,deque的考察
2. BST查询操作$O(log(N))$，单调队列为$O(1)$

## 参考

[huahua](https://zxi.mytechroad.com/blog/heap/leetcode-239-sliding-window-maximum/)